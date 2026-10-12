// Keep-alive HTTP load generator for content-object (media) requests.
// build: cc -O2 -pthread -o loadgen_media perf/loadgen_media.c
// usage: loadgen_media PORT BODY_FILE CONCURRENCY SECONDS
// BODY_FILE is a JSON request body for /v1/embeddings; every "@@ID@@" in it is
// replaced by a unique "r<client> n<seq>" string so the response cache never
// hits. Prints requests/s and latency p50/p90/p99 (ms).
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
static int port, secs;
static char *tmpl;
static size_t tmpl_len;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static long total;
static double *lat;
static size_t lat_n, lat_cap;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int conn(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (connect(fd, (struct sockaddr *)&a, sizeof a)) { perror("connect"); exit(1); }
    return fd;
}
static int write_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t k = write(fd, p, n);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}
static void *worker(void *arg) {
    long id = (long)arg, n = 0;
    size_t cap = tmpl_len + 4096;
    char *body = malloc(cap), *req = malloc(cap + 512), *buf = malloc(1 << 20);
    int fd = conn();
    double end = now() + secs;
    while (now() < end) {
        size_t p = 0;
        for (const char *s = tmpl; *s;) {
            const char *at = strstr(s, "@@ID@@");
            size_t run = at ? (size_t)(at - s) : strlen(s);
            memcpy(body + p, s, run); p += run; s += run;
            if (at) { p += (size_t)sprintf(body + p, "r%ld n%ld", id, n); s += 6; }
        }
        int h = sprintf(req, "POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", p);
        double t_start = now();
        if (write_all(fd, req, (size_t)h) || write_all(fd, body, p)) { close(fd); fd = conn(); continue; }
        size_t got = 0; long clen = -1; char *hdr_end = NULL;
        int closed = 0;
        for (;;) {
            ssize_t k = read(fd, buf + got, (1 << 20) - got - 1);
            if (k <= 0) { closed = 1; break; }
            got += (size_t)k; buf[got] = 0;
            if (!hdr_end && (hdr_end = strstr(buf, "\r\n\r\n"))) {
                char *cl = strcasestr(buf, "content-length:");
                clen = cl ? atol(cl + 15) : 0;
            }
            if (hdr_end && (long)(got - (size_t)(hdr_end + 4 - buf)) >= clen) break;
        }
        if (closed && !(hdr_end && (long)(got - (size_t)(hdr_end + 4 - buf)) >= clen)) {
            close(fd); fd = conn(); continue;
        }
        if (strncmp(buf, "HTTP/1.1 200", 12)) { fprintf(stderr, "bad response: %.200s\n", buf); exit(1); }
        n++;
        double ms = (now() - t_start) * 1e3;
        pthread_mutex_lock(&mu);
        if (lat_n == lat_cap) { lat_cap = lat_cap ? lat_cap * 2 : 4096; lat = realloc(lat, lat_cap * sizeof *lat); }
        lat[lat_n++] = ms;
        pthread_mutex_unlock(&mu);
        if (strcasestr(buf, "connection: close")) { close(fd); fd = conn(); }
    }
    pthread_mutex_lock(&mu); total += n; pthread_mutex_unlock(&mu);
    return NULL;
}
static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    if (argc != 5) { fprintf(stderr, "usage: %s PORT BODY_FILE CONCURRENCY SECONDS\n", argv[0]); return 2; }
    port = atoi(argv[1]); int c = atoi(argv[3]); secs = atoi(argv[4]);
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 2; }
    fseek(f, 0, SEEK_END); tmpl_len = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    tmpl = malloc(tmpl_len + 1);
    if (fread(tmpl, 1, tmpl_len, f) != tmpl_len) { perror("read"); return 2; }
    tmpl[tmpl_len] = 0; fclose(f);
    if (c < 1 || c > 4096) { fprintf(stderr, "concurrency must be 1..4096\n"); return 2; }
    pthread_t *t = malloc((size_t)c * sizeof *t);
    for (long i = 0; i < c; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < c; i++) pthread_join(t[i], NULL);
    qsort(lat, lat_n, sizeof *lat, cmp);
    double p50 = lat_n ? lat[lat_n / 2] : 0, p90 = lat_n ? lat[lat_n * 9 / 10] : 0,
           p99 = lat_n ? lat[lat_n * 99 / 100] : 0;
    printf("%.1f req/s  p50 %.2f ms  p90 %.2f ms  p99 %.2f ms  (n=%zu)\n", (double)total / secs, p50, p90, p99, lat_n);
    return 0;
}

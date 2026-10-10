// Minimal keep-alive HTTP load generator for embedding servers (no wrk/hey/ab needed).
// build: cc -O2 -pthread -o loadgen perf/loadgen.c
// usage: loadgen PORT WORDS CONCURRENCY SECONDS   (one token per filler word, plus ~12: 8170 words is ~8182 tokens)
// Prints requests/s and latency p50/p90/p99 (ms).
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
static int port, words, secs;
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
static void *worker(void *arg) {
    long id = (long)arg, n = 0;
    char *body = malloc(words * 24 + 128), *req = malloc(words * 24 + 512), *buf = malloc(1 << 20);
    int fd = conn();
    double end = now() + secs;
    while (now() < end) {
        /* Unique short prefix, then a fixed filler: token counts stay constant
         * across a run (numbered words grew tokens as counters grew). */
        int p = sprintf(body, "{\"model\":\"embeddinggemma-2\",\"input\":\"r%ld n%ld", id, n);
        for (int w = 0; w < words; w++) p += sprintf(body + p, " the");
        p += sprintf(body + p, "\"}");
        int r = sprintf(req, "POST /api/embed HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", p, body);
        double t_start = now();
        if (write(fd, req, r) != r) { close(fd); fd = conn(); continue; }
        size_t got = 0; long clen = -1; char *hdr_end = NULL;
        int closed = 0;
        for (;;) {
            ssize_t k = read(fd, buf + got, (1 << 20) - got - 1);
            if (k <= 0) { closed = 1; break; }
            got += k; buf[got] = 0;
            if (!hdr_end && (hdr_end = strstr(buf, "\r\n\r\n"))) {
                char *cl = strcasestr(buf, "content-length:");
                clen = cl ? atol(cl + 15) : 0;
            }
            if (hdr_end && (long)(got - (hdr_end + 4 - buf)) >= clen) break;
        }
        if (closed && !(hdr_end && (long)(got - (hdr_end + 4 - buf)) >= clen)) {
            close(fd); fd = conn(); continue; /* server ended keep-alive: retry */
        }
        if (strncmp(buf, "HTTP/1.1 200", 12)) { fprintf(stderr, "bad response: %.80s\n", buf); exit(1); }
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
int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    port = atoi(argv[1]), words = atoi(argv[2]); int c = atoi(argv[3]); secs = atoi(argv[4]);
    if (c < 1 || c > 4096) { fprintf(stderr, "concurrency must be 1..4096\n"); return 2; }
    pthread_t *t = malloc((size_t)c * sizeof *t);
    for (long i = 0; i < c; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < c; i++) pthread_join(t[i], NULL);
    int cmp(const void *a, const void *b);
    qsort(lat, lat_n, sizeof *lat, cmp);
    double p50 = lat_n ? lat[lat_n / 2] : 0, p90 = lat_n ? lat[lat_n * 9 / 10] : 0,
           p99 = lat_n ? lat[lat_n * 99 / 100] : 0;
    printf("%.1f req/s  p50 %.2f ms  p90 %.2f ms  p99 %.2f ms\n", (double)total / secs, p50, p90, p99);
    return 0;
}

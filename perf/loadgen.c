// Minimal keep-alive HTTP load generator for embedding servers (no wrk/hey/ab needed).
// build: cc -O2 -pthread -o loadgen perf/loadgen.c
// usage: loadgen PORT WORDS CONCURRENCY SECONDS   (~4.6 tokens per generated word)
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
        int p = sprintf(body, "{\"model\":\"embeddinggemma-2\",\"input\":\"");
        for (int w = 0; w < words; w++) p += sprintf(body + p, "w%ld%ld ", id, n * 31 + w);
        p += sprintf(body + p, "\"}");
        int r = sprintf(req, "POST /api/embed HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", p, body);
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
        if (strcasestr(buf, "connection: close")) { close(fd); fd = conn(); }
    }
    pthread_mutex_lock(&mu); total += n; pthread_mutex_unlock(&mu);
    return NULL;
}
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    port = atoi(argv[1]), words = atoi(argv[2]); int c = atoi(argv[3]); secs = atoi(argv[4]);
    pthread_t t[256];
    for (long i = 0; i < c; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < c; i++) pthread_join(t[i], NULL);
    printf("%.1f req/s\n", (double)total / secs);
    return 0;
}

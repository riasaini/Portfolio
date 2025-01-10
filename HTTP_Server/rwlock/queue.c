#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "queue.h"

typedef struct node {
    void *data;
    struct node *next;
} node_t;

struct queue {
    int size;
    node_t *head;
    node_t *tail;
    pthread_mutex_t lock;
    pthread_cond_t empty;
    pthread_cond_t full;
    int itemsInQueue;
};

static node_t *node_new(void *data) {
    node_t *node = (node_t *) malloc(sizeof(node_t));
    if (!node) {
        return NULL;
    }
    node->data = data;
    node->next = NULL;
    return node;
}

queue_t *queue_new(int size) {
    queue_t *queue = (queue_t *) malloc(sizeof(queue_t));
    if (!queue) {
        return NULL;
    }
    queue->size = size;
    queue->head = NULL;
    queue->tail = NULL;
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->empty, NULL);
    pthread_cond_init(&queue->full, NULL);
    queue->itemsInQueue = 0;
    return queue;
}

void queue_delete(queue_t **q) {
    if (!q || !*q) {
        return;
    }
    queue_t *queue = *q;
    node_t *node = queue->head;
    while (node) {
        node_t *tmp = node;
        node = node->next;
        free(tmp);
    }
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->empty);
    pthread_cond_destroy(&queue->full);
    free(queue);
    *q = NULL;
}

bool queue_push(queue_t *q, void *elem) {
    if (!q) {
        return false;
    }
    node_t *node = node_new(elem);
    if (!node) {
        return false;
    }
    pthread_mutex_lock(&q->lock);
    while (q->itemsInQueue == q->size) {
        pthread_cond_wait(&q->full, &q->lock);
    }
    if (!q->tail) {
        q->head = node;
        q->tail = node;
    } else {
        q->tail->next = node;
        q->tail = node;
    }
    q->itemsInQueue++;
    pthread_cond_signal(&q->empty);
    pthread_mutex_unlock(&q->lock);
    return true;
}

bool queue_pop(queue_t *q, void **elem) {
    if (!q || !elem) {
        return false;
    }
    pthread_mutex_lock(&q->lock);
    while (q->itemsInQueue == 0) {
        pthread_cond_wait(&q->empty, &q->lock);
    }
    node_t *node = q->head;
    *elem = node->data;
    q->head = node->next;
    if (!q->head) {
        q->tail = NULL;
    }
    free(node);
    q->itemsInQueue--;
    pthread_cond_signal(&q->full);
    pthread_mutex_unlock(&q->lock);
    return true;
}

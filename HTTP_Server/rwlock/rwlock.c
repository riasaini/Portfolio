#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "rwlock.h"

struct rwlock {
    pthread_mutex_t lock;
    pthread_cond_t readers;
    pthread_cond_t writers;
    int numReaders;
    int numWriters;
    int numWritersWaiting;
    int numReadersWaiting;
    PRIORITY p;
    int n;
    int r;
};

rwlock_t *rwlock_new(PRIORITY p, uint32_t n) {
    rwlock_t *newLock = (rwlock_t *) malloc(sizeof(rwlock_t));
    if (!newLock) {
        return NULL;
    }
    pthread_mutex_init(&newLock->lock, NULL);
    pthread_cond_init(&newLock->readers, NULL);
    pthread_cond_init(&newLock->writers, NULL);
    newLock->numReaders = 0;
    newLock->numWriters = 0;
    newLock->numWritersWaiting = 0;
    newLock->numReadersWaiting = 0;
    newLock->n = n;
    newLock->r = 0;
    newLock->p = p;
    return newLock;
}

void rwlock_delete(rwlock_t **rw) {
    if (!rw || !*rw) {
        return;
    }
    pthread_mutex_destroy(&(*rw)->lock);
    pthread_cond_destroy(&(*rw)->readers);
    pthread_cond_destroy(&(*rw)->writers);
    free(*rw);
    *rw = NULL;
}

void reader_lock(rwlock_t *rw) {
    pthread_mutex_lock(&rw->lock);
    rw->numReadersWaiting++;
    if (rw->p == READERS) {
        while (rw->numWriters > 0) {
            pthread_cond_wait(&rw->readers, &rw->lock);
        }
    } else if (rw->p == WRITERS) {
        while (rw->numWriters > 0 || rw->numWritersWaiting > 0) {
            pthread_cond_wait(&rw->readers, &rw->lock);
        }
    } else if (rw->p == N_WAY) {
        while (rw->numWriters > 0 || (rw->r >= rw->n && rw->numWritersWaiting > 0)) {
            pthread_cond_wait(&rw->readers, &rw->lock);
        }
        rw->r++;
    }
    rw->numReadersWaiting--;
    rw->numReaders++;
    pthread_mutex_unlock(&rw->lock);
}

void reader_unlock(rwlock_t *rw) {
    pthread_mutex_lock(&rw->lock);
    rw->numReaders--;
    if (rw->p == READERS) {
        if (rw->numReaders == 0 && rw->numWritersWaiting > 0 && rw->numReadersWaiting == 0
            && rw->numWriters == 0) {
            pthread_cond_signal(&rw->writers);
        }
    } else if (rw->p == WRITERS) {
        if (rw->numReaders == 0 && rw->numWritersWaiting > 0 && rw->numWriters == 0) {
            pthread_cond_signal(&rw->writers);
        }
    } else if (rw->p == N_WAY) {
        if (rw->r >= rw->n && rw->numWritersWaiting > 0 && rw->numReaders == 0) {
            pthread_cond_signal(&rw->writers);
        } else if (rw->r < rw->n && rw->numReadersWaiting == 0 && rw->numReaders == 0
                   && rw->numWritersWaiting > 0) {
            pthread_cond_signal(&rw->writers);
        }
    }
    pthread_mutex_unlock(&rw->lock);
}

void writer_lock(rwlock_t *rw) {
    pthread_mutex_lock(&rw->lock);
    rw->numWritersWaiting++;
    if (rw->p == READERS) {
        while (rw->numWriters > 0 || rw->numReaders > 0 || rw->numReadersWaiting > 0) {
            pthread_cond_wait(&rw->writers, &rw->lock);
        }
    } else if (rw->p == WRITERS) {
        while (rw->numWriters > 0 || rw->numReaders > 0) {
            pthread_cond_wait(&rw->writers, &rw->lock);
        }
    } else if (rw->p == N_WAY) {
        while ((rw->r < rw->n && rw->numReadersWaiting > 0) || rw->numReaders > 0
               || rw->numWriters > 0) {
            pthread_cond_wait(&rw->writers, &rw->lock);
        }
        rw->r = 0;
    }
    rw->numWritersWaiting--;
    rw->numWriters++;
    pthread_mutex_unlock(&rw->lock);
}

void writer_unlock(rwlock_t *rw) {
    pthread_mutex_lock(&rw->lock);
    rw->numWriters--;
    if (rw->p == READERS) {
        if (rw->numReadersWaiting > 0) {
            pthread_cond_broadcast(&rw->readers);
        } else if (rw->numWritersWaiting > 0 && rw->numReaders == 0 && rw->numReadersWaiting == 0
                   && rw->numWriters == 0) {
            pthread_cond_signal(&rw->writers);
        }
    } else if (rw->p == WRITERS) {
        if (rw->numWritersWaiting > 0) {
            pthread_cond_signal(&rw->writers);
        } else if (rw->numWritersWaiting == 0 && rw->numWriters == 0 && rw->numReadersWaiting > 0) {
            pthread_cond_broadcast(&rw->readers);
        }
    } else if (rw->p == N_WAY) {
        if (rw->numReadersWaiting > 0 && rw->r < rw->n) {
            pthread_cond_broadcast(&rw->readers);
        } else if (rw->numReadersWaiting == 0 && rw->numWritersWaiting > 0) {
            pthread_cond_signal(&rw->writers);
        }
    }
    pthread_mutex_unlock(&rw->lock);
}

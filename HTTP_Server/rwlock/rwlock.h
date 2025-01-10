#pragma once

#include <stdint.h>

/** @struct rwlock_t
 *
 *  @brief This typedef renames the struct rwlock.
 */
typedef struct rwlock rwlock_t;

typedef enum { READERS, WRITERS, N_WAY } PRIORITY;

/** @brief Dynamically allocates and initializes a new rwlock with
 *         priority p, and, if using N_WAY priority, n.
 *
 *  @param The priority of the rwlock
 *
 *  @param The n value, if using N_WAY priority
 *
 *  @return a pointer to a new rwlock_t
 */

rwlock_t *rwlock_new(PRIORITY p, uint32_t n);

/** @brief Delete rwlock and free all of its memory.
 *
 *  @param rw the rwlock to be deleted.
 *
 */
void rwlock_delete(rwlock_t **rw);

/** @brief acquire rw for reading
 *
 */
void reader_lock(rwlock_t *rw);

/** @brief release rw for reading--the thread
 * releasing the lock has already acquired it for reading.
 *
 */
void reader_unlock(rwlock_t *rw);

/** @brief acquire rw for writing
 *
 */
void writer_lock(rwlock_t *rw);

/** @brief release rw for writing--the thread
 * releasing the lock has already acquired it for writing.
 *
 */
void writer_unlock(rwlock_t *rw);

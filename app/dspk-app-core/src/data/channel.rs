//! A queue from worker threads to the UI thread (finished frames, progress, readouts).
//!
//! The UI thread awaits it; workers put values in without waiting. No lock is involved: in a
//! browser the page's main thread may not wait for one, and a channel that takes a lock on both
//! sides (as `async-channel` does) stops the page the moment a worker and the page touch it
//! together.

use futures_util::StreamExt as _;

/// The sending half: cheap to clone, usable from any thread.
pub struct Sender<T>(futures_channel::mpsc::UnboundedSender<T>);

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// The receiving half. The queue is closed for it.
pub struct Receiver<T>(futures_channel::mpsc::UnboundedReceiver<T>);

/// The other half is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed;

/// A queue without a size limit.
pub fn unbounded<T>() -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = futures_channel::mpsc::unbounded();
    (Sender(tx), Receiver(rx))
}

impl<T> Sender<T> {
    /// Puts `value` in the queue (never waits).
    pub fn try_send(&self, value: T) -> Result<(), Closed> {
        self.0.unbounded_send(value).map_err(|_| Closed)
    }

    /// Closes the queue: the receiver gets what is in it, then [`Closed`].
    pub fn close(&self) {
        self.0.close_channel();
    }
}

impl<T> Receiver<T> {
    /// The next value; [`Closed`] once every sender is gone and the queue is empty.
    pub async fn recv(&mut self) -> Result<T, Closed> {
        self.0.next().await.ok_or(Closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_cross_threads_in_order_and_the_end_is_seen() {
        let (tx, mut rx) = unbounded::<u32>();
        let worker = std::thread::spawn(move || (0..100).for_each(|i| tx.try_send(i).unwrap()));
        worker.join().unwrap();
        let got: Vec<u32> = futures_executor_lite(async move {
            let mut got = Vec::new();
            while let Ok(v) = rx.recv().await {
                got.push(v);
            }
            got
        });
        assert_eq!(got, (0..100).collect::<Vec<_>>());
    }

    /// Runs a future that never has to wait for another thread.
    fn futures_executor_lite<T>(future: impl std::future::Future<Output = T>) -> T {
        let mut future = std::pin::pin!(future);
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        loop {
            if let std::task::Poll::Ready(v) = future.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }
}

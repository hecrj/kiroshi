pub use futures::SinkExt;
pub use futures::stream::{Stream, StreamExt};

use futures::Future;
use futures::channel::mpsc;
use futures::stream;

pub fn from_future<T, E, F>(
    f: impl FnOnce(mpsc::Sender<T>) -> F,
) -> impl Stream<Item = Result<T, E>>
where
    F: Future<Output = Result<(), E>>,
{
    let (sender, receiver) = mpsc::channel(1);

    stream::select(
        receiver.map(Ok),
        stream::once(f(sender)).filter_map(|result| async move {
            match result {
                Ok(()) => None,
                Err(error) => Some(Err(error)),
            }
        }),
    )
}

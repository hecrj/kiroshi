pub use plug::{Connection, Never, Plug, Strip};

use std::io;

const ADDRESS: &str = "127.0.0.1:9149";
pub const PORT: u64 = 9149;

pub async fn connect<I, O>(plug: Plug<I, O>) -> io::Result<Connection<I, O>> {
    plug.connect(ADDRESS).await
}

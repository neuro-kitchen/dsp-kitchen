//! Length-prefixed messages on QUIC streams: a [`LENGTH_PREFIX_BYTES`]-byte big-endian length,
//! then the protobuf encoding.

use prost::Message;

use super::LENGTH_PREFIX_BYTES;
use crate::error::{StreamError, StreamResult};

/// Writes `message` with its length prefix.
pub async fn write_message(stream: &mut quinn::SendStream, message: &impl Message) -> StreamResult<()> {
    let len = message.encoded_len();
    let prefix = u32::try_from(len).map_err(|_| StreamError::TooLarge { len, max: u32::MAX as usize })?;
    let mut buf = Vec::with_capacity(LENGTH_PREFIX_BYTES + len);
    buf.extend_from_slice(&prefix.to_be_bytes());
    message.encode(&mut buf).expect("the buffer holds the encoded length");
    stream.write_all(&buf).await?;
    Ok(())
}

/// Reads the next message. `Ok(None)` when the stream ended cleanly before a message; an error
/// when it ended inside one or the length exceeds `max_bytes` (checked before allocating).
pub async fn read_message<M: Message + Default>(stream: &mut quinn::RecvStream, max_bytes: usize) -> StreamResult<Option<M>> {
    let mut prefix = [0u8; LENGTH_PREFIX_BYTES];
    match stream.read_exact(&mut prefix).await {
        Ok(()) => {}
        Err(quinn::ReadExactError::FinishedEarly(0)) => return Ok(None),
        Err(quinn::ReadExactError::FinishedEarly(_)) => return Err(StreamError::Truncated),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(prefix) as usize;
    if len > max_bytes {
        return Err(StreamError::TooLarge { len, max: max_bytes });
    }
    let mut buf = vec![0u8; len];
    match stream.read_exact(&mut buf).await {
        Ok(()) => Ok(Some(M::decode(buf.as_slice())?)),
        Err(quinn::ReadExactError::FinishedEarly(_)) => Err(StreamError::Truncated),
        Err(e) => Err(e.into()),
    }
}

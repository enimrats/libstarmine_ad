use thiserror::Error;

const MAJOR_SYNC_FBA: [u8; 4] = [0xF8, 0x72, 0x6F, 0xBA];
const MAJOR_SYNC_FBB: [u8; 4] = [0xF8, 0x72, 0x6F, 0xBB];

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RawTrueHdError {
    #[error("no-major-sync offset={offset}")]
    NoMajorSync { offset: usize },
    #[error("invalid-access-unit-length offset={offset} length={length}")]
    InvalidAccessUnitLength { offset: usize, length: usize },
    #[error("truncated-access-unit offset={offset} expected={expected} available={available}")]
    TruncatedAccessUnit {
        offset: usize,
        expected: usize,
        available: usize,
    },
}

pub struct RawTrueHdAccessUnit<'a> {
    pub offset: usize,
    pub bytes: &'a [u8],
}

pub struct RawTrueHdAccessUnitIter<'a> {
    data: &'a [u8],
    offset: usize,
    synced: bool,
}

impl<'a> RawTrueHdAccessUnitIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            offset: 0,
            synced: false,
        }
    }

    fn find_initial_access_unit_start(&self) -> Option<usize> {
        let search_start = self.offset.saturating_add(4).max(4);
        if search_start + 4 > self.data.len() {
            return None;
        }

        self.data[search_start..]
            .windows(4)
            .position(|window| window == MAJOR_SYNC_FBA || window == MAJOR_SYNC_FBB)
            .map(|index| search_start + index - 4)
    }
}

impl<'a> Iterator for RawTrueHdAccessUnitIter<'a> {
    type Item = Result<RawTrueHdAccessUnit<'a>, RawTrueHdError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        if !self.synced {
            let Some(start) = self.find_initial_access_unit_start() else {
                let offset = self.offset;
                self.offset = self.data.len();
                return Some(Err(RawTrueHdError::NoMajorSync { offset }));
            };
            self.offset = start;
            self.synced = true;
        }

        if self.offset + 2 > self.data.len() {
            let offset = self.offset;
            let available = self.data.len().saturating_sub(offset);
            self.offset = self.data.len();
            return Some(Err(RawTrueHdError::TruncatedAccessUnit {
                offset,
                expected: 2,
                available,
            }));
        }

        let length = ((u16::from_be_bytes([self.data[self.offset], self.data[self.offset + 1]])
            & 0x0FFF) as usize)
            << 1;
        if length < 8 {
            let offset = self.offset;
            self.offset = self.data.len();
            return Some(Err(RawTrueHdError::InvalidAccessUnitLength {
                offset,
                length,
            }));
        }

        let available = self.data.len() - self.offset;
        if length > available {
            let offset = self.offset;
            self.offset = self.data.len();
            return Some(Err(RawTrueHdError::TruncatedAccessUnit {
                offset,
                expected: length,
                available,
            }));
        }

        let access_unit = RawTrueHdAccessUnit {
            offset: self.offset,
            bytes: &self.data[self.offset..self.offset + length],
        };
        self.offset += length;
        Some(Ok(access_unit))
    }
}

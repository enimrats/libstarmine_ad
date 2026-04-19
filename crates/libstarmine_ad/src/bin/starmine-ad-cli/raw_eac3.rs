use starmine_ad::{ParseError, inspect_access_unit};

pub struct RawEac3Frame<'a> {
    pub offset: usize,
    pub bytes: &'a [u8],
}

pub struct RawEac3FrameIter<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> RawEac3FrameIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for RawEac3FrameIter<'a> {
    type Item = Result<RawEac3Frame<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        match inspect_access_unit(&self.data[self.offset..]) {
            Ok(info) => {
                let available = self.data.len() - self.offset;
                if info.frame_size > available {
                    let err = ParseError::TruncatedFrame {
                        expected: info.frame_size,
                        available,
                    };
                    self.offset = self.data.len();
                    return Some(Err(err));
                }

                let frame = RawEac3Frame {
                    offset: self.offset,
                    bytes: &self.data[self.offset..self.offset + info.frame_size],
                };
                self.offset += info.frame_size;
                Some(Ok(frame))
            }
            Err(err) => {
                self.offset = self.data.len();
                Some(Err(err))
            }
        }
    }
}

use std::sync::OnceLock;

pub(crate) const QMF_SUBBANDS: usize = 64;
const QMF_DOUBLE_LENGTH: usize = QMF_SUBBANDS * 2;
const QMF_COEFFS_LEN: usize = 640;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QmfSubbands {
    pub real: [f32; QMF_SUBBANDS],
    pub imaginary: [f32; QMF_SUBBANDS],
}

impl QmfSubbands {
    pub fn zero() -> Self {
        Self {
            real: [0.0; QMF_SUBBANDS],
            imaginary: [0.0; QMF_SUBBANDS],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct QuadratureMirrorFilterBank {
    input_stream_forward: [f32; QMF_COEFFS_LEN],
    input_stream_forward_head: usize,
    input_stream_inverse: [f32; QMF_COEFFS_LEN * 2],
    input_stream_inverse_head: usize,
}

impl QuadratureMirrorFilterBank {
    pub fn new() -> Self {
        Self {
            input_stream_forward: [0.0; QMF_COEFFS_LEN],
            input_stream_forward_head: 0,
            input_stream_inverse: [0.0; QMF_COEFFS_LEN * 2],
            input_stream_inverse_head: 0,
        }
    }

    pub fn process_forward(&mut self, input: &[f32]) -> QmfSubbands {
        debug_assert_eq!(input.len(), QMF_SUBBANDS);

        self.input_stream_forward_head =
            wrap_ring_head(self.input_stream_forward_head, QMF_SUBBANDS, QMF_COEFFS_LEN);
        for (offset, sample) in input.iter().rev().enumerate() {
            let slot = ring_index(self.input_stream_forward_head, offset, QMF_COEFFS_LEN);
            self.input_stream_forward[slot] = *sample;
        }

        let mut grouping = [0.0f32; QMF_DOUBLE_LENGTH];
        for (sample, slot) in grouping.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let mut source = sample;
            while source < QMF_COEFFS_LEN {
                sum += self.input_stream_forward
                    [ring_index(self.input_stream_forward_head, source, QMF_COEFFS_LEN)]
                    * QMF_COEFFS[source];
                source += QMF_DOUBLE_LENGTH;
            }
            *slot = sum;
        }

        let cache = qmf_cache();
        let mut result = QmfSubbands::zero();
        for subband in 0..QMF_SUBBANDS {
            result.real[subband] = dot(&cache.forward_real[subband], &grouping);
            result.imaginary[subband] = dot(&cache.forward_imaginary[subband], &grouping);
        }
        result
    }

    pub fn process_inverse(&mut self, input: &QmfSubbands, output: &mut [f32]) {
        debug_assert_eq!(output.len(), QMF_SUBBANDS);

        let inverse_len = self.input_stream_inverse.len();
        self.input_stream_inverse_head = wrap_ring_head(
            self.input_stream_inverse_head,
            QMF_DOUBLE_LENGTH,
            inverse_len,
        );

        let cache = qmf_cache();
        for sample in 0..QMF_DOUBLE_LENGTH {
            let mut value = 0.0f32;
            for subband in 0..QMF_SUBBANDS {
                value += cache.inverse_real[subband][sample] * input.real[subband];
                value -= cache.inverse_imaginary[subband][sample] * input.imaginary[subband];
            }
            let slot = ring_index(self.input_stream_inverse_head, sample, inverse_len);
            self.input_stream_inverse[slot] = value;
        }

        output.fill(0.0);
        for sample in 0..QMF_SUBBANDS {
            output[sample] = self.input_stream_inverse
                [ring_index(self.input_stream_inverse_head, sample, inverse_len)]
                * QMF_COEFFS[sample]
                + self.input_stream_inverse[ring_index(
                    self.input_stream_inverse_head,
                    QMF_SUBBANDS * 3 + sample,
                    inverse_len,
                )] * QMF_COEFFS[QMF_SUBBANDS + sample];
        }
        for group in 1..(QMF_COEFFS_LEN / QMF_DOUBLE_LENGTH) {
            let time_slot = QMF_SUBBANDS * 4 * group;
            let coeff_slot = QMF_DOUBLE_LENGTH * group;
            let time_pair = time_slot + QMF_SUBBANDS * 3;
            let coeff_pair = coeff_slot + QMF_SUBBANDS;
            for sample in 0..QMF_SUBBANDS {
                output[sample] += self.input_stream_inverse[ring_index(
                    self.input_stream_inverse_head,
                    time_slot + sample,
                    inverse_len,
                )] * QMF_COEFFS[coeff_slot + sample]
                    + self.input_stream_inverse[ring_index(
                        self.input_stream_inverse_head,
                        time_pair + sample,
                        inverse_len,
                    )] * QMF_COEFFS[coeff_pair + sample];
            }
        }
    }
}

impl Default for QuadratureMirrorFilterBank {
    fn default() -> Self {
        Self::new()
    }
}

fn dot(lhs: &[f32; QMF_DOUBLE_LENGTH], rhs: &[f32; QMF_DOUBLE_LENGTH]) -> f32 {
    let mut sum = 0.0f32;
    for index in 0..QMF_DOUBLE_LENGTH {
        sum += lhs[index] * rhs[index];
    }
    sum
}

#[inline]
fn wrap_ring_head(head: usize, step: usize, len: usize) -> usize {
    if head >= step {
        head - step
    } else {
        len + head - step
    }
}

#[inline]
fn ring_index(head: usize, offset: usize, len: usize) -> usize {
    let index = head + offset;
    if index >= len { index - len } else { index }
}

struct QmfCache {
    forward_real: [[f32; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
    forward_imaginary: [[f32; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
    inverse_real: [[f32; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
    inverse_imaginary: [[f32; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
}

fn qmf_cache() -> &'static QmfCache {
    static CACHE: OnceLock<QmfCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut cache = QmfCache {
            forward_real: [[0.0; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
            forward_imaginary: [[0.0; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
            inverse_real: [[0.0; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
            inverse_imaginary: [[0.0; QMF_DOUBLE_LENGTH]; QMF_SUBBANDS],
        };
        let subband_div = 1.0f32 / QMF_SUBBANDS as f32;
        for subband in 0..QMF_SUBBANDS {
            for sample in 0..QMF_DOUBLE_LENGTH {
                let exp = std::f32::consts::PI
                    * (subband as f32 + 0.5)
                    * (sample as f32 - 0.5)
                    * subband_div;
                cache.forward_real[subband][sample] = exp.cos();
                cache.forward_imaginary[subband][sample] = exp.sin();

                let inverse_exp = std::f32::consts::PI
                    * (subband as f32 + 0.5)
                    * (sample as f32 - QMF_DOUBLE_LENGTH as f32 + 0.5)
                    * subband_div;
                cache.inverse_real[subband][sample] = inverse_exp.cos() * subband_div;
                cache.inverse_imaginary[subband][sample] = inverse_exp.sin() * subband_div;
            }
        }
        cache
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReferenceQmfBank {
        input_stream_forward: [f32; QMF_COEFFS_LEN],
        input_stream_inverse: [f32; QMF_COEFFS_LEN * 2],
    }

    impl ReferenceQmfBank {
        fn new() -> Self {
            Self {
                input_stream_forward: [0.0; QMF_COEFFS_LEN],
                input_stream_inverse: [0.0; QMF_COEFFS_LEN * 2],
            }
        }

        fn process_forward(&mut self, input: &[f32]) -> QmfSubbands {
            self.input_stream_forward
                .copy_within(0..QMF_COEFFS_LEN - QMF_SUBBANDS, QMF_SUBBANDS);
            for (slot, sample) in self
                .input_stream_forward
                .iter_mut()
                .take(QMF_SUBBANDS)
                .zip(input.iter().rev())
            {
                *slot = *sample;
            }

            let mut grouping = [0.0f32; QMF_DOUBLE_LENGTH];
            for (sample, slot) in grouping.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                let mut source = sample;
                while source < QMF_COEFFS_LEN {
                    sum += self.input_stream_forward[source] * QMF_COEFFS[source];
                    source += QMF_DOUBLE_LENGTH;
                }
                *slot = sum;
            }

            let cache = qmf_cache();
            let mut result = QmfSubbands::zero();
            for subband in 0..QMF_SUBBANDS {
                result.real[subband] = dot(&cache.forward_real[subband], &grouping);
                result.imaginary[subband] = dot(&cache.forward_imaginary[subband], &grouping);
            }
            result
        }

        fn process_inverse(&mut self, input: &QmfSubbands, output: &mut [f32]) {
            let inverse_len = self.input_stream_inverse.len();
            self.input_stream_inverse
                .copy_within(0..inverse_len - QMF_DOUBLE_LENGTH, QMF_DOUBLE_LENGTH);

            let cache = qmf_cache();
            for sample in 0..QMF_DOUBLE_LENGTH {
                let mut value = 0.0f32;
                for subband in 0..QMF_SUBBANDS {
                    value += cache.inverse_real[subband][sample] * input.real[subband];
                    value -= cache.inverse_imaginary[subband][sample] * input.imaginary[subband];
                }
                self.input_stream_inverse[sample] = value;
            }

            output.fill(0.0);
            for sample in 0..QMF_SUBBANDS {
                output[sample] = self.input_stream_inverse[sample] * QMF_COEFFS[sample]
                    + self.input_stream_inverse[QMF_SUBBANDS * 3 + sample]
                        * QMF_COEFFS[QMF_SUBBANDS + sample];
            }
            for group in 1..(QMF_COEFFS_LEN / QMF_DOUBLE_LENGTH) {
                let time_slot = QMF_SUBBANDS * 4 * group;
                let coeff_slot = QMF_DOUBLE_LENGTH * group;
                let time_pair = time_slot + QMF_SUBBANDS * 3;
                let coeff_pair = coeff_slot + QMF_SUBBANDS;
                for sample in 0..QMF_SUBBANDS {
                    output[sample] += self.input_stream_inverse[time_slot + sample]
                        * QMF_COEFFS[coeff_slot + sample]
                        + self.input_stream_inverse[time_pair + sample]
                            * QMF_COEFFS[coeff_pair + sample];
                }
            }
        }
    }

    #[test]
    fn forward_ring_buffer_matches_copy_within_reference() {
        let mut actual = QuadratureMirrorFilterBank::new();
        let mut reference = ReferenceQmfBank::new();
        let mut seed = 0x1234_5678;

        for _ in 0..64 {
            let input = next_block(&mut seed);
            let actual_subbands = actual.process_forward(&input);
            let reference_subbands = reference.process_forward(&input);
            assert_subbands_close(&actual_subbands, &reference_subbands);
        }
    }

    #[test]
    fn inverse_ring_buffer_matches_copy_within_reference() {
        let mut actual = QuadratureMirrorFilterBank::new();
        let mut reference = ReferenceQmfBank::new();
        let mut seed = 0x8765_4321;

        for _ in 0..64 {
            let input = next_subbands(&mut seed);
            let mut actual_output = [0.0f32; QMF_SUBBANDS];
            let mut reference_output = [0.0f32; QMF_SUBBANDS];
            actual.process_inverse(&input, &mut actual_output);
            reference.process_inverse(&input, &mut reference_output);
            assert_output_close(&actual_output, &reference_output);
        }
    }

    fn next_block(seed: &mut u32) -> [f32; QMF_SUBBANDS] {
        let mut output = [0.0f32; QMF_SUBBANDS];
        for sample in &mut output {
            *sample = next_sample(seed);
        }
        output
    }

    fn next_subbands(seed: &mut u32) -> QmfSubbands {
        let mut output = QmfSubbands::zero();
        for sample in &mut output.real {
            *sample = next_sample(seed);
        }
        for sample in &mut output.imaginary {
            *sample = next_sample(seed);
        }
        output
    }

    fn next_sample(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let normalized = (*seed >> 8) as f32 / ((1u32 << 24) as f32);
        normalized * 2.0 - 1.0
    }

    fn assert_subbands_close(actual: &QmfSubbands, expected: &QmfSubbands) {
        for index in 0..QMF_SUBBANDS {
            assert!(
                (actual.real[index] - expected.real[index]).abs() <= f32::EPSILON,
                "real[{index}] mismatch: {} vs {}",
                actual.real[index],
                expected.real[index],
            );
            assert!(
                (actual.imaginary[index] - expected.imaginary[index]).abs() <= f32::EPSILON,
                "imaginary[{index}] mismatch: {} vs {}",
                actual.imaginary[index],
                expected.imaginary[index],
            );
        }
    }

    fn assert_output_close(actual: &[f32; QMF_SUBBANDS], expected: &[f32; QMF_SUBBANDS]) {
        for index in 0..QMF_SUBBANDS {
            assert!(
                (actual[index] - expected[index]).abs() <= f32::EPSILON,
                "output[{index}] mismatch: {} vs {}",
                actual[index],
                expected[index],
            );
        }
    }
}

const QMF_COEFFS: [f32; QMF_COEFFS_LEN] = [
    0.000000000000000e+000f32,
    1.990318758627504e-004f32,
    2.494762615491542e-004f32,
    3.021769445225078e-004f32,
    3.548460080857985e-004f32,
    4.058915811480806e-004f32,
    4.546408052001889e-004f32,
    5.012680176678405e-004f32,
    5.464958142195282e-004f32,
    5.912073950641334e-004f32,
    6.361178026937039e-004f32,
    6.816060488244358e-004f32,
    7.277257095064290e-004f32,
    7.743418255606097e-004f32,
    8.212990636826637e-004f32,
    8.685363488152327e-004f32,
    9.161071539925993e-004f32,
    9.641168291303352e-004f32,
    1.012630507392736e-003f32,
    1.061605258108620e-003f32,
    1.110882587090581e-003f32,
    1.160236901298543e-003f32,
    1.209448942573337e-003f32,
    1.258362795150757e-003f32,
    1.306902381715039e-003f32,
    1.355046337751365e-003f32,
    1.402784629568410e-003f32,
    1.450086694843816e-003f32,
    1.496898951224534e-003f32,
    1.543170821958483e-003f32,
    1.588889089195869e-003f32,
    1.634098242730728e-003f32,
    1.678892372493930e-003f32,
    1.723381173920660e-003f32,
    1.767651163797991e-003f32,
    1.811741998614740e-003f32,
    1.855650606587200e-003f32,
    1.899360915083620e-003f32,
    1.942876625831283e-003f32,
    1.986241654706626e-003f32,
    2.029534125962055e-003f32,
    2.072840712410525e-003f32,
    2.116229103721749e-003f32,
    2.159738034390673e-003f32,
    2.203392976200947e-003f32,
    2.247239773881968e-003f32,
    2.291373966775394e-003f32,
    2.335946110021889e-003f32,
    2.381132815654862e-003f32,
    2.427086732976290e-003f32,
    2.473891839822582e-003f32,
    2.521550367974952e-003f32,
    2.570013995199655e-003f32,
    2.619244058999978e-003f32,
    2.669265893796866e-003f32,
    2.720177146231281e-003f32,
    2.772088849679780e-003f32,
    2.825009494162980e-003f32,
    2.878716544061140e-003f32,
    2.932677076291194e-003f32,
    2.986067366389476e-003f32,
    3.037905983043366e-003f32,
    3.087269477594307e-003f32,
    3.133519274378684e-003f32,
    3.176460810085721e-003f32,
    3.216374095471449e-003f32,
    3.253902493849856e-003f32,
    3.289837867273167e-003f32,
    3.324873276103132e-003f32,
    3.359407689115599e-003f32,
    3.393454084675361e-003f32,
    3.426668323773391e-003f32,
    3.458465815999750e-003f32,
    3.488171121469781e-003f32,
    3.515141351338780e-003f32,
    3.538827383683883e-003f32,
    3.558767785536742e-003f32,
    3.574539247363964e-003f32,
    3.585697968628984e-003f32,
    3.591743339500398e-003f32,
    3.592116764752254e-003f32,
    3.586228204993297e-003f32,
    3.573492966885132e-003f32,
    3.553356715665694e-003f32,
    3.525300399274114e-003f32,
    3.488824092931520e-003f32,
    3.443423145747434e-003f32,
    3.388568319085867e-003f32,
    3.323699442173841e-003f32,
    3.248231770523395e-003f32,
    3.161568930730635e-003f32,
    3.063113666967670e-003f32,
    2.952270973359112e-003f32,
    2.828441943181057e-003f32,
    2.691016173288500e-003f32,
    2.539366102140493e-003f32,
    2.372848583221744e-003f32,
    2.190814088754598e-003f32,
    1.992618085548526e-003f32,
    1.777631090142623e-003f32,
    1.545242163079598e-003f32,
    1.294855985911958e-003f32,
    1.025885587325796e-003f32,
    7.377456851538827e-004f32,
    4.298496740962311e-004f32,
    1.016113723823784e-004f32,
    -2.475493814535340e-004f32,
    -6.181972580227641e-004f32,
    -1.010876063031582e-003f32,
    -1.426108207321696e-003f32,
    -1.864392667409557e-003f32,
    -2.326207721179968e-003f32,
    -2.812013688448634e-003f32,
    -3.322252633537029e-003f32,
    -3.857344314546718e-003f32,
    -4.417678415707104e-003f32,
    -5.003604409245843e-003f32,
    -5.615422427540850e-003f32,
    -6.253382198869787e-003f32,
    -6.917691380307223e-003f32,
    -7.608536937561301e-003f32,
    -8.326113472848559e-003f32,
    -9.070651572928327e-003f32,
    -9.842433610911637e-003f32,
    -1.064178450184536e-002f32,
    -1.146903570409307e-002f32,
    -1.232446526717138e-002f32,
    -1.320822893615923e-002f32,
    1.412030102138547e-002f32,
    1.506045143737221e-002f32,
    1.602824700934038e-002f32,
    1.702310507234504e-002f32,
    1.804435938034114e-002f32,
    1.909132707403387e-002f32,
    2.016335321815832e-002f32,
    2.125982139139435e-002f32,
    2.238013015948307e-002f32,
    2.352365148441367e-002f32,
    2.468968228813486e-002f32,
    2.587741357605385e-002f32,
    2.708591966384863e-002f32,
    2.831416731612567e-002f32,
    2.956103453432552e-002f32,
    3.082532788511644e-002f32,
    3.210578787607558e-002f32,
    3.340108247607704e-002f32,
    3.470979250147262e-002f32,
    3.603039785904666e-002f32,
    3.736126987823528e-002f32,
    3.870067428980750e-002f32,
    4.004677994303860e-002f32,
    4.139766786359423e-002f32,
    4.275134353925827e-002f32,
    4.410572893128047e-002f32,
    4.545866171224587e-002f32,
    4.680788921400311e-002f32,
    4.815106534667384e-002f32,
    4.948575188369231e-002f32,
    5.080942296260306e-002f32,
    5.211947012173918e-002f32,
    5.341320372603929e-002f32,
    5.468785186395163e-002f32,
    5.594055607104873e-002f32,
    5.716836923188953e-002f32,
    5.836825629443718e-002f32,
    5.953709945765930e-002f32,
    6.067170625396996e-002f32,
    6.176881705202805e-002f32,
    6.282510999827461e-002f32,
    6.383720245755561e-002f32,
    6.480165083585107e-002f32,
    6.571495100350305e-002f32,
    6.657354346196487e-002f32,
    6.737381445564891e-002f32,
    6.811211000439976e-002f32,
    6.878473991370719e-002f32,
    6.938797895654626e-002f32,
    6.991806618580000e-002f32,
    7.037120381110623e-002f32,
    7.074355866301176e-002f32,
    7.103126866531538e-002f32,
    7.123045563399449e-002f32,
    7.133723888151840e-002f32,
    7.134774334517399e-002f32,
    7.125810128129656e-002f32,
    7.106444395777428e-002f32,
    7.076288963679085e-002f32,
    7.034953453342756e-002f32,
    6.982045490146145e-002f32,
    6.917172452383333e-002f32,
    6.839944399575645e-002f32,
    6.749977716975542e-002f32,
    6.646898181809889e-002f32,
    6.530342654389224e-002f32,
    6.399958984339946e-002f32,
    6.255404354954748e-002f32,
    6.096342863203985e-002f32,
    5.922443337469448e-002f32,
    5.733378365410422e-002f32,
    5.528824660015738e-002f32,
    5.308464739461209e-002f32,
    5.071989148277166e-002f32,
    4.819098634672628e-002f32,
    4.549505579582869e-002f32,
    4.262934676625042e-002f32,
    3.959122947020497e-002f32,
    3.637819581239452e-002f32,
    3.298786054608736e-002f32,
    2.941796954479800e-002f32,
    2.566640058060906e-002f32,
    2.173117939155709e-002f32,
    1.761048656968719e-002f32,
    1.330266415707108e-002f32,
    8.806217289921706e-003f32,
    4.119815918461287e-003f32,
    -7.577038291607129e-004f32,
    -5.827337082489678e-003f32,
    -1.108990619665782e-002f32,
    -1.654605559674886e-002f32,
    -2.219624707735291e-002f32,
    -2.804075556277473e-002f32,
    -3.407966641908426e-002f32,
    -4.031287253355741e-002f32,
    -4.674007190475649e-002f32,
    -5.336076390182971e-002f32,
    -6.017424526940620e-002f32,
    -6.717960594283154e-002f32,
    -7.437572538762392e-002f32,
    -8.176127022450692e-002f32,
    -8.933469320120192e-002f32,
    -9.709423309043450e-002f32,
    -1.050379143754414e-001f32,
    -1.131635475471188e-001f32,
    -1.214687284677367e-001f32,
    -1.299508386078101e-001f32,
    -1.386070430802319e-001f32,
    -1.474342913196958e-001f32,
    -1.564293167898782e-001f32,
    -1.655886374953163e-001f32,
    -1.749085568711785e-001f32,
    -1.843851642116290e-001f32,
    -1.940143360850268e-001f32,
    -2.037917371113644e-001f32,
    -2.137128217101543e-001f32,
    -2.237728356363325e-001f32,
    -2.339668182208061e-001f32,
    -2.442896055908444e-001f32,
    -2.547358344658102e-001f32,
    -2.652999476893712e-001f32,
    -2.759762003673840e-001f32,
    -2.867586659726799e-001f32,
    -2.976412485679301e-001f32,
    -3.086176827721830e-001f32,
    -3.196815399704708e-001f32,
    -3.308262316588501e-001f32,
    -3.420450091826495e-001f32,
    3.533309414505971e-001f32,
    3.646770149404552e-001f32,
    3.760759747758828e-001f32,
    3.875204555118187e-001f32,
    3.990029533969267e-001f32,
    4.105158411581483e-001f32,
    4.220513789540003e-001f32,
    4.336017251305980e-001f32,
    4.451589452332786e-001f32,
    4.567150149423557e-001f32,
    4.682618290579831e-001f32,
    4.797912086537587e-001f32,
    4.912949058677955e-001f32,
    5.027646134968753e-001f32,
    5.141919746376279e-001f32,
    5.255685924518015e-001f32,
    5.368860394090674e-001f32,
    5.481358656081351e-001f32,
    5.593096071830315e-001f32,
    5.703987947306394e-001f32,
    5.813949615434598e-001f32,
    5.922896536434017e-001f32,
    6.030744392774144e-001f32,
    6.137409201916185e-001f32,
    6.242807411441345e-001f32,
    6.346855991963545e-001f32,
    6.449472531836600e-001f32,
    6.550575323798634e-001f32,
    6.650083455855346e-001f32,
    6.747916901830467e-001f32,
    6.843996616799759e-001f32,
    6.938244627003839e-001f32,
    7.030584122393319e-001f32,
    7.120939537241190e-001f32,
    7.209236637533725e-001f32,
    7.295402599029810e-001f32,
    7.379366091028713e-001f32,
    7.461057359576386e-001f32,
    7.540408314942230e-001f32,
    7.617352611504460e-001f32,
    7.691825714586890e-001f32,
    7.763765020733762e-001f32,
    7.833109874824341e-001f32,
    7.899801646390305e-001f32,
    7.963783815797485e-001f32,
    8.025002033685581e-001f32,
    8.083404191294724e-001f32,
    8.138940486031526e-001f32,
    8.191563476989879e-001f32,
    8.241228138607196e-001f32,
    8.287891904413357e-001f32,
    8.331514714928793e-001f32,
    8.372059062705359e-001f32,
    8.409490040631689e-001f32,
    8.443775395556067e-001f32,
    8.474885573145614e-001f32,
    8.502793750759253e-001f32,
    8.527475863595390e-001f32,
    8.548910606594570e-001f32,
    8.567079441260879e-001f32,
    8.581966597760032e-001f32,
    8.593559096378087e-001f32,
    8.601846769933608e-001f32,
    8.606822313166693e-001f32,
    8.608481078185764e-001f32,
    8.606822313166693e-001f32,
    8.601846769933608e-001f32,
    8.593559096378087e-001f32,
    8.581966597760032e-001f32,
    8.567079441260879e-001f32,
    8.548910606594570e-001f32,
    8.527475863595390e-001f32,
    8.502793750759253e-001f32,
    8.474885573145614e-001f32,
    8.443775395556067e-001f32,
    8.409490040631689e-001f32,
    8.372059062705359e-001f32,
    8.331514714928793e-001f32,
    8.287891904413357e-001f32,
    8.241228138607196e-001f32,
    8.191563476989879e-001f32,
    8.138940486031526e-001f32,
    8.083404191294724e-001f32,
    8.025002033685581e-001f32,
    7.963783815797485e-001f32,
    7.899801646390305e-001f32,
    7.833109874824341e-001f32,
    7.763765020733762e-001f32,
    7.691825714586890e-001f32,
    7.617352611504460e-001f32,
    7.540408314942230e-001f32,
    7.461057359576386e-001f32,
    7.379366091028713e-001f32,
    7.295402599029810e-001f32,
    7.209236637533725e-001f32,
    7.120939537241190e-001f32,
    7.030584122393319e-001f32,
    6.938244627003839e-001f32,
    6.843996616799759e-001f32,
    6.747916901830467e-001f32,
    6.650083455855346e-001f32,
    6.550575323798634e-001f32,
    6.449472531836600e-001f32,
    6.346855991963545e-001f32,
    6.242807411441345e-001f32,
    6.137409201916185e-001f32,
    6.030744392774144e-001f32,
    5.922896536434017e-001f32,
    5.813949615434598e-001f32,
    5.703987947306394e-001f32,
    5.593096071830315e-001f32,
    5.481358656081351e-001f32,
    5.368860394090674e-001f32,
    5.255685924518015e-001f32,
    5.141919746376279e-001f32,
    5.027646134968753e-001f32,
    4.912949058677955e-001f32,
    4.797912086537587e-001f32,
    4.682618290579831e-001f32,
    4.567150149423557e-001f32,
    4.451589452332786e-001f32,
    4.336017251305980e-001f32,
    4.220513789540003e-001f32,
    4.105158411581483e-001f32,
    3.990029533969267e-001f32,
    3.875204555118187e-001f32,
    3.760759747758828e-001f32,
    3.646770149404552e-001f32,
    -3.533309414505971e-001f32,
    -3.420450091826495e-001f32,
    -3.308262316588501e-001f32,
    -3.196815399704708e-001f32,
    -3.086176827721830e-001f32,
    -2.976412485679301e-001f32,
    -2.867586659726799e-001f32,
    -2.759762003673840e-001f32,
    -2.652999476893712e-001f32,
    -2.547358344658102e-001f32,
    -2.442896055908444e-001f32,
    -2.339668182208061e-001f32,
    -2.237728356363325e-001f32,
    -2.137128217101543e-001f32,
    -2.037917371113644e-001f32,
    -1.940143360850268e-001f32,
    -1.843851642116290e-001f32,
    -1.749085568711785e-001f32,
    -1.655886374953163e-001f32,
    -1.564293167898782e-001f32,
    -1.474342913196958e-001f32,
    -1.386070430802319e-001f32,
    -1.299508386078101e-001f32,
    -1.214687284677367e-001f32,
    -1.131635475471188e-001f32,
    -1.050379143754414e-001f32,
    -9.709423309043450e-002f32,
    -8.933469320120192e-002f32,
    -8.176127022450692e-002f32,
    -7.437572538762392e-002f32,
    -6.717960594283154e-002f32,
    -6.017424526940620e-002f32,
    -5.336076390182971e-002f32,
    -4.674007190475649e-002f32,
    -4.031287253355741e-002f32,
    -3.407966641908426e-002f32,
    -2.804075556277473e-002f32,
    -2.219624707735291e-002f32,
    -1.654605559674886e-002f32,
    -1.108990619665782e-002f32,
    -5.827337082489678e-003f32,
    -7.577038291607129e-004f32,
    4.119815918461287e-003f32,
    8.806217289921706e-003f32,
    1.330266415707108e-002f32,
    1.761048656968719e-002f32,
    2.173117939155709e-002f32,
    2.566640058060906e-002f32,
    2.941796954479800e-002f32,
    3.298786054608736e-002f32,
    3.637819581239452e-002f32,
    3.959122947020497e-002f32,
    4.262934676625042e-002f32,
    4.549505579582869e-002f32,
    4.819098634672628e-002f32,
    5.071989148277166e-002f32,
    5.308464739461209e-002f32,
    5.528824660015738e-002f32,
    5.733378365410422e-002f32,
    5.922443337469448e-002f32,
    6.096342863203985e-002f32,
    6.255404354954748e-002f32,
    6.399958984339946e-002f32,
    6.530342654389224e-002f32,
    6.646898181809889e-002f32,
    6.749977716975542e-002f32,
    6.839944399575645e-002f32,
    6.917172452383333e-002f32,
    6.982045490146145e-002f32,
    7.034953453342756e-002f32,
    7.076288963679085e-002f32,
    7.106444395777428e-002f32,
    7.125810128129656e-002f32,
    7.134774334517399e-002f32,
    7.133723888151840e-002f32,
    7.123045563399449e-002f32,
    7.103126866531538e-002f32,
    7.074355866301176e-002f32,
    7.037120381110623e-002f32,
    6.991806618580000e-002f32,
    6.938797895654626e-002f32,
    6.878473991370719e-002f32,
    6.811211000439976e-002f32,
    6.737381445564891e-002f32,
    6.657354346196487e-002f32,
    6.571495100350305e-002f32,
    6.480165083585107e-002f32,
    6.383720245755561e-002f32,
    6.282510999827461e-002f32,
    6.176881705202805e-002f32,
    6.067170625396996e-002f32,
    5.953709945765930e-002f32,
    5.836825629443718e-002f32,
    5.716836923188953e-002f32,
    5.594055607104873e-002f32,
    5.468785186395163e-002f32,
    5.341320372603929e-002f32,
    5.211947012173918e-002f32,
    5.080942296260306e-002f32,
    4.948575188369231e-002f32,
    4.815106534667384e-002f32,
    4.680788921400311e-002f32,
    4.545866171224587e-002f32,
    4.410572893128047e-002f32,
    4.275134353925827e-002f32,
    4.139766786359423e-002f32,
    4.004677994303860e-002f32,
    3.870067428980750e-002f32,
    3.736126987823528e-002f32,
    3.603039785904666e-002f32,
    3.470979250147262e-002f32,
    3.340108247607704e-002f32,
    3.210578787607558e-002f32,
    3.082532788511644e-002f32,
    2.956103453432552e-002f32,
    2.831416731612567e-002f32,
    2.708591966384863e-002f32,
    2.587741357605385e-002f32,
    2.468968228813486e-002f32,
    2.352365148441367e-002f32,
    2.238013015948307e-002f32,
    2.125982139139435e-002f32,
    2.016335321815832e-002f32,
    1.909132707403387e-002f32,
    1.804435938034114e-002f32,
    1.702310507234504e-002f32,
    1.602824700934038e-002f32,
    1.506045143737221e-002f32,
    -1.412030102138547e-002f32,
    -1.320822893615923e-002f32,
    -1.232446526717138e-002f32,
    -1.146903570409307e-002f32,
    -1.064178450184536e-002f32,
    -9.842433610911637e-003f32,
    -9.070651572928327e-003f32,
    -8.326113472848559e-003f32,
    -7.608536937561301e-003f32,
    -6.917691380307223e-003f32,
    -6.253382198869787e-003f32,
    -5.615422427540850e-003f32,
    -5.003604409245843e-003f32,
    -4.417678415707104e-003f32,
    -3.857344314546718e-003f32,
    -3.322252633537029e-003f32,
    -2.812013688448634e-003f32,
    -2.326207721179968e-003f32,
    -1.864392667409557e-003f32,
    -1.426108207321696e-003f32,
    -1.010876063031582e-003f32,
    -6.181972580227641e-004f32,
    -2.475493814535340e-004f32,
    1.016113723823784e-004f32,
    4.298496740962311e-004f32,
    7.377456851538827e-004f32,
    1.025885587325796e-003f32,
    1.294855985911958e-003f32,
    1.545242163079598e-003f32,
    1.777631090142623e-003f32,
    1.992618085548526e-003f32,
    2.190814088754598e-003f32,
    2.372848583221744e-003f32,
    2.539366102140493e-003f32,
    2.691016173288500e-003f32,
    2.828441943181057e-003f32,
    2.952270973359112e-003f32,
    3.063113666967670e-003f32,
    3.161568930730635e-003f32,
    3.248231770523395e-003f32,
    3.323699442173841e-003f32,
    3.388568319085867e-003f32,
    3.443423145747434e-003f32,
    3.488824092931520e-003f32,
    3.525300399274114e-003f32,
    3.553356715665694e-003f32,
    3.573492966885132e-003f32,
    3.586228204993297e-003f32,
    3.592116764752254e-003f32,
    3.591743339500398e-003f32,
    3.585697968628984e-003f32,
    3.574539247363964e-003f32,
    3.558767785536742e-003f32,
    3.538827383683883e-003f32,
    3.515141351338780e-003f32,
    3.488171121469781e-003f32,
    3.458465815999750e-003f32,
    3.426668323773391e-003f32,
    3.393454084675361e-003f32,
    3.359407689115599e-003f32,
    3.324873276103132e-003f32,
    3.289837867273167e-003f32,
    3.253902493849856e-003f32,
    3.216374095471449e-003f32,
    3.176460810085721e-003f32,
    3.133519274378684e-003f32,
    3.087269477594307e-003f32,
    3.037905983043366e-003f32,
    2.986067366389476e-003f32,
    2.932677076291194e-003f32,
    2.878716544061140e-003f32,
    2.825009494162980e-003f32,
    2.772088849679780e-003f32,
    2.720177146231281e-003f32,
    2.669265893796866e-003f32,
    2.619244058999978e-003f32,
    2.570013995199655e-003f32,
    2.521550367974952e-003f32,
    2.473891839822582e-003f32,
    2.427086732976290e-003f32,
    2.381132815654862e-003f32,
    2.335946110021889e-003f32,
    2.291373966775394e-003f32,
    2.247239773881968e-003f32,
    2.203392976200947e-003f32,
    2.159738034390673e-003f32,
    2.116229103721749e-003f32,
    2.072840712410525e-003f32,
    2.029534125962055e-003f32,
    1.986241654706626e-003f32,
    1.942876625831283e-003f32,
    1.899360915083620e-003f32,
    1.855650606587200e-003f32,
    1.811741998614740e-003f32,
    1.767651163797991e-003f32,
    1.723381173920660e-003f32,
    1.678892372493930e-003f32,
    1.634098242730728e-003f32,
    1.588889089195869e-003f32,
    1.543170821958483e-003f32,
    1.496898951224534e-003f32,
    1.450086694843816e-003f32,
    1.402784629568410e-003f32,
    1.355046337751365e-003f32,
    1.306902381715039e-003f32,
    1.258362795150757e-003f32,
    1.209448942573337e-003f32,
    1.160236901298543e-003f32,
    1.110882587090581e-003f32,
    1.061605258108620e-003f32,
    1.012630507392736e-003f32,
    9.641168291303352e-004f32,
    9.161071539925993e-004f32,
    8.685363488152327e-004f32,
    8.212990636826637e-004f32,
    7.743418255606097e-004f32,
    7.277257095064290e-004f32,
    6.816060488244358e-004f32,
    6.361178026937039e-004f32,
    5.912073950641334e-004f32,
    5.464958142195282e-004f32,
    5.012680176678405e-004f32,
    4.546408052001889e-004f32,
    4.058915811480806e-004f32,
    3.548460080857985e-004f32,
    3.021769445225078e-004f32,
    2.494762615491542e-004f32,
    1.990318758627504e-004f32,
];

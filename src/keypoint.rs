#[derive(Debug, Clone)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub octave: usize,
    pub orientation: f32,
    pub descriptor: Vec<f32>,
}

impl KeyPoint {
    pub fn new(
        x: f32,
        y: f32,
        scale: f32,
        octave: usize,
        orientation: f32,
        descriptor: Vec<f32>
      ) -> Self {
        KeyPoint {
            x: x,
            y: y,
            scale: scale,
            octave: octave,
            orientation: orientation,
            descriptor: descriptor
        }
    }
}
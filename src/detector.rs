use crate::keypoint::KeyPoint;
use nalgebra::{Matrix3, Vector3, LU};
use ndarray::{Array3, Axis};

const CONTRAST_THRESHOLD: f32 = 0.03;
const CURVATURE_THRESHOLD: f32 = 10.0;
const INITIAL_SIGMA: f32 = 1.6;
const SCALES_PER_OCTAVE: usize = 5;

/// SIFT detector for finding keypoints in images
pub struct SiftDetector {
    pub scale_spaces: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>>,
    num_octaves: usize,
    num_scales: usize,
    sigma: f32,
}

impl SiftDetector {
    pub fn new() -> Self {
        SiftDetector {
            scale_spaces: Vec::new(),
            num_octaves: 4,
            num_scales: 5,
            sigma: 1.6,
        }
    }
    pub fn detect(&mut self, image: image::ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Vec<KeyPoint> {
        // let mut scale_spaces: Vec<
        //     ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>,
        // > = Vec::new();
        let mut dog_pyramids: Vec<
            ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>,
        > = Vec::new();
        let mut img_first = image.clone();
        for octave in 0..self.num_octaves {
            let mut n_width;
            let mut n_height;
            (n_width, n_height) = img_first.dimensions();
            if (octave > 0) {
                let ii = self.scale_spaces[octave - 1].index_axis(Axis(0), 2);
                let img_temp =
                    image::ImageBuffer::from_fn(n_width as u32, n_height as u32, |x, y| {
                        let val = (ii[[y as usize, x as usize]].clamp(0.0, 1.0) * 255.0) as u8;
                        image::Luma::<u8>([val])
                    });
                let (w, h) = img_temp.dimensions();
                img_first = image::imageops::resize(
                    &img_temp,
                    w / 2,
                    h / 2,
                    image::imageops::FilterType::Nearest,
                );
                (n_width, n_height) = img_first.dimensions();
            }
            print!(
                "Image resized successfully. New width: {}, New height: {}\n",
                n_width, n_height
            );
            let mut gaussian_pyramid =
                Array3::<f32>::zeros((self.num_scales, n_height as usize, n_width as usize));
            for scale in 0..self.num_scales {
                let k = self.sigma
                    * 2.0_f32.powf(octave as f32 + scale as f32 / self.num_scales as f32);
                print!("Building Gaussian pyramid for scale {}, k {}\n", scale, k);
                let kernel_size = (6.0 * k) as u32;
                let blurred_img = image::imageops::blur(&img_first, kernel_size as f32);
                for (x, y, pixel) in blurred_img.enumerate_pixels() {
                    gaussian_pyramid[(scale, y as usize, x as usize)] = pixel[0] as f32;
                }
            }
            // println!("{:?}", gaussian_pyramid);
            let mut dog_pyramid =
                Array3::<f32>::zeros((self.num_scales - 1, n_height as usize, n_width as usize));
            for scale in 1..self.num_scales {
                let a_channel = gaussian_pyramid.index_axis(Axis(0), scale);
                let b_channel = gaussian_pyramid.index_axis(Axis(0), scale - 1);
                let mut result_channel = dog_pyramid.index_axis_mut(Axis(0), scale - 1);
                result_channel.assign(&(&a_channel - &b_channel));
            }
            self.scale_spaces.push(gaussian_pyramid);
            dog_pyramids.push(dog_pyramid);
        }
        // println!("{:?}", v);
        let mut keypoints = Vec::new();
        for (octave, dog_pyramid) in dog_pyramids.iter().enumerate() {
            let (scales, w, h) = dog_pyramid.dim();
            println!(
                "Octave {}, width {}, height {}, depth {}",
                octave, scales, w, h
            );
            for scale in 1..scales - 1 {
                let prev = dog_pyramid.index_axis(Axis(0), scale - 1);
                let current = dog_pyramid.index_axis(Axis(0), scale);
                let next = dog_pyramid.index_axis(Axis(0), scale + 1);
                for y in 1..current.shape()[0] - 1 {
                    for x in 1..current.shape()[1] - 1 {
                        let center = current[[y, x]];
                        if self.is_extremum(prev, current, next, x, y, center) {
                            if let Some((dx, dy, ds, value)) =
                                self.refine_extremum(prev, current, next, x, y)
                            {
                                // println!("Extremum found at dx {}, dy {}, ds {}, value {}", dx, dy, ds, value);
                                if value.abs() < CONTRAST_THRESHOLD {
                                    continue;
                                }

                                // 消除边缘响应
                                if self.is_edge_response(current, x, y) {
                                    continue;
                                }

                                let scale = INITIAL_SIGMA
                                    * 2f32.powf(
                                        (octave as f32
                                            + (scale as f32 + ds) / SCALES_PER_OCTAVE as f32),
                                    );

                                keypoints.push(KeyPoint {
                                    x: (x as f32 + dx) * (1 << octave) as f32,
                                    y: (y as f32 + dy) * (1 << octave) as f32,
                                    scale,
                                    octave,
                                    orientation: 0.0, // 后续计算方向
                                    descriptor: Vec::new(),
                                });
                            }
                        }
                    }
                }
            }
        }
        keypoints
    }

    fn is_extremum(
        &self,
        prev: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        current: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        next: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        x: usize,
        y: usize,
        center: f32,
    ) -> bool {
        // 检查当前层
        for dy in -1..=1 {
            for dx in -1..=1 {
                if current[[(y as isize + dy) as usize, (x as isize + dx) as usize]] > center {
                    return false;
                }
            }
        }

        // // 检查上层和下层
        // for dz in -1..=1 {
        //     let layer = if dz == -1 { prev } else if dz == 0 { current } else { next };
        //     for dy in -1..=1 {
        //         for dx in -1..=1 {
        //             if layer[[(y as isize + dy) as usize, (x as isize + dx) as usize]] > center {
        //                 return false;
        //             }
        //         }
        //     }
        // }
        true
    }

    fn refine_extremum(
        &self,
        prev: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        current: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        next: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        x: usize,
        y: usize,
    ) -> Option<(f32, f32, f32, f32)> {
        const MAX_ITER: usize = 5;
        const CONVERGE_THRESH: f32 = 0.01;
        const MAX_SHIFT: f32 = 0.6;

        let mut xi = Vector3::new(0., 0., 0.); // 偏移量 (dx, dy, ds)
        let mut iter = 0;
        let (width, height) = (current.shape()[1], current.shape()[0]);

        // 获取三维邻域数据（当前点周围3x3x3立方体）
        let mut samples = Array3::<f32>::zeros((3, 3, 3));
        let mut grad = Vector3::new(0., 0., 0.);
        loop {
            // 检查边界
            if x as f32 + xi[0] < 0.
                || x as f32 + xi[0] >= width as f32
                || y as f32 + xi[1] < 0.
                || y as f32 + xi[1] >= height as f32
            {
                return None;
            }
            samples[[0, 0, 0]] = prev[[x - 1, y - 1]];
            samples[[0, 1, 0]] = prev[[x - 1, y]];
            samples[[0, 2, 0]] = prev[[x - 1, y + 1]];
            samples[[1, 0, 0]] = prev[[x, y - 1]];
            samples[[1, 1, 0]] = prev[[x, y]];
            samples[[1, 2, 0]] = prev[[x, y + 1]];
            samples[[2, 0, 0]] = prev[[x + 1, y - 1]];
            samples[[2, 1, 0]] = prev[[x + 1, y]];
            samples[[2, 2, 0]] = prev[[x + 1, y + 1]];

            samples[[0, 0, 1]] = current[[x - 1, y - 1]];
            samples[[0, 1, 1]] = current[[x - 1, y]];
            samples[[0, 2, 1]] = current[[x - 1, y + 1]];
            samples[[1, 0, 1]] = current[[x, y - 1]];
            samples[[1, 1, 1]] = current[[x, y]];
            samples[[1, 2, 1]] = current[[x, y + 1]];
            samples[[2, 0, 1]] = current[[x + 1, y - 1]];
            samples[[2, 1, 1]] = current[[x + 1, y]];
            samples[[2, 2, 1]] = current[[x + 1, y + 1]];

            samples[[0, 0, 2]] = next[[x - 1, y - 1]];
            samples[[0, 1, 2]] = next[[x - 1, y]];
            samples[[0, 2, 2]] = next[[x - 1, y + 1]];
            samples[[1, 0, 2]] = next[[x, y - 1]];
            samples[[1, 1, 2]] = next[[x, y]];
            samples[[1, 2, 2]] = next[[x, y + 1]];
            samples[[2, 0, 2]] = next[[x + 1, y - 1]];
            samples[[2, 1, 2]] = next[[x + 1, y]];
            samples[[2, 2, 2]] = next[[x + 1, y + 1]];

            // 计算一阶导数（梯度）
            let dx = (samples[[1, 1, 2]] - samples[[1, 1, 0]]) / 2.0;
            let dy = (samples[[1, 2, 1]] - samples[[1, 0, 1]]) / 2.0;
            let ds = (samples[[2, 1, 1]] - samples[[0, 1, 1]]) / 2.0;

            grad[0] = dx;
            grad[1] = dy;
            grad[2] = ds;

            // 计算二阶导数（Hessian矩阵）
            let dxx = samples[[1, 1, 2]] - 2.0 * samples[[1, 1, 1]] + samples[[1, 1, 0]];
            let dyy = samples[[1, 2, 1]] - 2.0 * samples[[1, 1, 1]] + samples[[1, 0, 1]];
            let dss = samples[[2, 1, 1]] - 2.0 * samples[[1, 1, 1]] + samples[[0, 1, 1]];
            let dxy = (samples[[1, 2, 2]] - samples[[1, 2, 0]] - samples[[1, 0, 2]]
                + samples[[1, 0, 0]])
                / 4.0;
            let dxs = (samples[[2, 1, 2]] - samples[[2, 1, 0]] - samples[[0, 1, 2]]
                + samples[[0, 1, 0]])
                / 4.0;
            let dys = (samples[[2, 2, 1]] - samples[[2, 0, 1]] - samples[[0, 2, 1]]
                + samples[[0, 0, 1]])
                / 4.0;

            let hessian = Matrix3::new(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);

            // 解线性方程：Hessian * x = -grad
            let lu = LU::new(hessian);
            if lu.determinant().abs() < 1e-6 {
                return None; // Hessian不可逆
            }
            let delta = lu.solve(&(-grad)).unwrap();

            // 更新偏移量
            xi += delta;

            // 收敛判断
            if delta.norm() < CONVERGE_THRESH || iter >= MAX_ITER {
                break;
            }
            iter += 1;
        }

        // 最终偏移量检查
        if xi[0] > MAX_SHIFT || xi[1] > MAX_SHIFT || xi[2] > 1.0 {
            return None;
        }

        // 计算调整后的极值强度
        let d = samples[[1, 1, 1]] + 0.5 * grad.dot(&xi);
        Some((xi[0], xi[1], xi[2], d))
    }

    fn is_edge_response(
        &self,
        image: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        x: usize,
        y: usize,
    ) -> bool {
        let dxx = image[[y, x + 1]] + image[[y, x - 1]] - 2.0 * image[[y, x]];
        let dyy = image[[y + 1, x]] + image[[y - 1, x]] - 2.0 * image[[y, x]];
        let dxy = (image[[y + 1, x + 1]] - image[[y + 1, x - 1]] - image[[y - 1, x + 1]]
            + image[[y - 1, x - 1]])
            / 4.0;

        let trace = dxx + dyy;
        let det = dxx * dyy - dxy * dxy;

        (trace * trace) / det > (CURVATURE_THRESHOLD + 1.0).powi(2) / CURVATURE_THRESHOLD
    }
}

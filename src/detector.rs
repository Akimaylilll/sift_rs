use crate::gaussian_blur::GaussianFilter;
use crate::keypoint::KeyPoint;
use image::DynamicImage;
use nalgebra::{Matrix3, Vector3, LU};
use ndarray::{Array2, Array3, Axis};

const CONTRAST_THRESHOLD: f32 = 0.04;
const CURVATURE_THRESHOLD: f32 = 10.0;

/// SIFT detector for finding keypoints in images
pub struct SiftDetector {
    pub scale_spaces: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>>,
    num_scales: usize,
    num_octaves: usize,
    sigma: f32,
}

impl SiftDetector {
    pub fn new() -> Self {
        SiftDetector {
            scale_spaces: Vec::new(),
            num_scales: 3,
            num_octaves: 0,
            sigma: 1.6,
        }
    }
    pub fn detect(&mut self, image: image::DynamicImage) -> Vec<KeyPoint> {
        let image = self.dynamic_image_to_ndarray(&image);
        let mut dog_pyramids: Vec<
            ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>,
        > = Vec::new();
        let (image_width, image_height) = (image.shape()[1], image.shape()[0]);
        // 计算八度数
        let mut image_size = image_width.min(image_height);
        let mut num_octaves = 0;
        while image_size >= 8 {
            // 至少需要8x8的图像来检测关键点
            num_octaves += 1;
            image_size /= 2;
        }
        self.num_octaves = num_octaves;
        let n_dog = self.num_scales + 2; // DOG金字塔层数，s+2
        let n_g = self.num_scales + 3; // 高斯金字塔层数，s+3

        let mut current_img = image.clone();
        for octave in 0..self.num_octaves {
            let (w, h) = if octave == 0 {
                (image_width, image_height)
            } else {
                (current_img.shape()[1], current_img.shape()[0])
            };
            let mut gaussian_pyramid = Array3::<f32>::zeros((n_g, h as usize, w as usize));
            for scale in 0..n_g {
                // 计算每个尺度下的sigma值
                let k = 2.0_f32.powf(scale as f32 / self.num_scales as f32);
                let sigma = self.sigma * k;
                let blurred_img =
                    GaussianFilter::apply_to_array_view(current_img.view(), sigma as f64);
                gaussian_pyramid
                    .slice_mut(ndarray::s![scale, .., ..])
                    .assign(&blurred_img);
            }

            let mut dog_pyramid = Array3::<f32>::zeros((n_dog, h as usize, w as usize)); // DOG金字塔有n_dog层
            for scale in 1..n_g {
                let a_channel = gaussian_pyramid.index_axis(Axis(0), scale);
                let b_channel = gaussian_pyramid.index_axis(Axis(0), scale - 1);
                let mut result_channel = dog_pyramid.index_axis_mut(Axis(0), scale - 1);
                result_channel.assign(&(&a_channel - &b_channel));
            }
            self.scale_spaces.push(gaussian_pyramid);
            dog_pyramids.push(dog_pyramid);

            // 缩放到下一个八度
            if octave < self.num_octaves - 1 {
                // 获取当前八度的下采样源图像（通常是高斯金字塔中尺度翻倍的层）
                let downsample_source_idx = self.num_scales; // 参考sift.rs，取第s层进行下采样
                if downsample_source_idx < self.scale_spaces[octave].len_of(Axis(0)) {
                    current_img = self.scale_spaces[octave]
                        .index_axis(Axis(0), downsample_source_idx)
                        .to_owned();
                    current_img = self.resize_nearest_grayscale_simple(&current_img, h / 2, w / 2);
                } else {
                    // 如果索引超出了范围，则使用最后一层
                    current_img = self.scale_spaces[octave]
                        .index_axis(Axis(0), self.num_scales - 1)
                        .to_owned();
                    current_img = self.resize_nearest_grayscale_simple(&current_img, h / 2, w / 2);
                }
                // 下采样后sigma需要除以2，但下一八度会重新计算sigma，所以这里不显式调整
            }
        }

        let mut keypoints: Vec<KeyPoint> = Vec::new();
        for (octave, dog_pyramid) in dog_pyramids.iter().enumerate() {
            let (scales, _w, _h) = dog_pyramid.dim();
            for scale in 1..scales - 1 {
                let prev = dog_pyramid.index_axis(Axis(0), scale - 1);
                let current = dog_pyramid.index_axis(Axis(0), scale);
                let next = dog_pyramid.index_axis(Axis(0), scale + 1);
                for y in 1..current.shape()[0] - 1 {
                    for x in 1..current.shape()[1] - 1 {
                        let center = current[[y, x]];
                        if self.is_extremum(prev, current, next, x, y, center, CONTRAST_THRESHOLD) {
                            if let Some((dx, dy, _ds, value)) =
                                self.refine_extremum(prev, current, next, x, y, scale)
                            {
                                // 检查偏移量是否过大（表示插值不可靠）
                                if dx.abs() >= 1.0 || dy.abs() >= 1.0 || _ds.abs() >= 1.0 {
                                    continue; // 偏移量过大，跳过这个点
                                }
                                if value.abs() < CONTRAST_THRESHOLD {
                                    continue;
                                }

                                // 消除边缘响应
                                if self.is_edge_response(current, x, y) {
                                    continue;
                                }

                                // 计算实际尺度值：scale * 2^(octave + scale_offset/num_scales)
                                let scale_factor = 2.0_f32.powf(
                                    octave as f32 + (scale as f32 + _ds) / self.num_scales as f32,
                                );
                                let actual_scale = self.sigma * scale_factor;

                                keypoints.push(KeyPoint {
                                    x: (x as f32 + dx) * (1 << octave) as f32,
                                    y: (y as f32 + dy) * (1 << octave) as f32,
                                    // 正确计算实际尺度
                                    scale: actual_scale,
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
        threshold: f32,
    ) -> bool {
        if center.abs() < threshold {
            return false;
        }

        let mut is_local_maximum = true;
        let mut is_local_minimum = true;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (dx, dy, dz) = (dx as isize, dy as isize, dz as isize);
                    let (current_x, current_y) = (x as isize, y as isize);

                    // 跳过中心点
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }

                    let (check_x, check_y) = ((current_x + dx) as usize, (current_y + dy) as usize);

                    // 根据dz选择层
                    let val = if dz == -1 {
                        // 检查prev层
                        if check_x >= prev.shape()[1] || check_y >= prev.shape()[0] {
                            continue;
                        }
                        prev[[check_y, check_x]]
                    } else if dz == 0 {
                        // 检查current层
                        if check_x >= current.shape()[1] || check_y >= current.shape()[0] {
                            continue;
                        }
                        current[[check_y, check_x]]
                    } else if dz == 1 {
                        // 检查next层
                        if check_x >= next.shape()[1] || check_y >= next.shape()[0] {
                            continue;
                        }
                        next[[check_y, check_x]]
                    } else {
                        continue;
                    };

                    // 更新最大最小值判断
                    if val >= center {
                        is_local_maximum = false;
                    } else if val <= center {
                        is_local_minimum = false;
                    }
                }
            }
        }

        is_local_maximum || is_local_minimum
    }

    fn refine_extremum(
        &self,
        prev: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        current: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        next: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        x: usize,
        y: usize,
        scale: usize,
    ) -> Option<(f32, f32, f32, f32)> {
        const MAX_ITER: usize = 5;
        const CONVERGE_THRESH: f32 = 0.5;

        let mut offset = Vector3::new(0., 0., 0.);
        let (width, height) = (current.shape()[1], current.shape()[0]);
        let dog_scales = self.num_scales + 2;

        let mut x_float = x as f32;
        let mut y_float = y as f32;
        let mut s_float = scale as f32;

        for _iter in 0..MAX_ITER {
            let x_int = x_float as usize;
            let y_int = y_float as usize;
            let s_int = s_float as usize;

            // 检查边界 - DOG金字塔的边界检查
            if x_int < 1
                || x_int >= width - 1
                || y_int < 1
                || y_int >= height - 1
                || s_int < 1
                || s_int >= dog_scales - 1
            {
                return None;
            }

            // 计算一阶导数（梯度）
            let dx = (current[[y_int, x_int + 1]] - current[[y_int, x_int - 1]]) * 0.5;
            let dy = (current[[y_int + 1, x_int]] - current[[y_int - 1, x_int]]) * 0.5;
            let ds = (next[[y_int, x_int]] - prev[[y_int, x_int]]) * 0.5;

            // 计算二阶导数（Hessian矩阵）
            let dxx = current[[y_int, x_int + 1]] - 2.0 * current[[y_int, x_int]]
                + current[[y_int, x_int - 1]];
            let dyy = current[[y_int + 1, x_int]] - 2.0 * current[[y_int, x_int]]
                + current[[y_int - 1, x_int]];
            let dss = next[[y_int, x_int]] - 2.0 * current[[y_int, x_int]] + prev[[y_int, x_int]];

            let dxy = (current[[y_int + 1, x_int + 1]]
                - current[[y_int + 1, x_int - 1]]
                - current[[y_int - 1, x_int + 1]]
                + current[[y_int - 1, x_int - 1]])
                * 0.25;
            let dxs =
                (next[[y_int, x_int + 1]] - next[[y_int, x_int - 1]] - prev[[y_int, x_int + 1]]
                    + prev[[y_int, x_int - 1]])
                    * 0.25;
            let dys =
                (next[[y_int + 1, x_int]] - next[[y_int - 1, x_int]] - prev[[y_int + 1, x_int]]
                    + prev[[y_int - 1, x_int]])
                    * 0.25;

            let hessian = Matrix3::new(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);

            let lu = LU::new(hessian);
            if lu.determinant().abs() < 1e-6 {
                return None; // Hessian不可逆
            }

            let grad = Vector3::new(dx, dy, ds);
            let delta = lu.solve(&(-grad)).unwrap();

            // 更新偏移量
            offset += delta;

            // 更新坐标
            x_float += delta[0];
            y_float += delta[1];
            s_float += delta[2];

            // 收敛判断
            if delta.norm() < CONVERGE_THRESH {
                // 检查最终位置是否在边界内
                if x_float < 0.0
                    || x_float >= (width as f32)
                    || y_float < 0.0
                    || y_float >= (height as f32)
                    || s_float < 0.0
                    || s_float >= dog_scales as f32
                {
                    return None;
                }

                // 计算在极值点的函数值（使用完整的二阶泰勒展开）
                let dog_value = current[[y_int, x_int]]
                    + grad.dot(&offset)
                    + 0.5 * (offset.transpose() * hessian * offset)[0];

                // 返回相对于当前点的偏移量
                return Some((delta[0], delta[1], delta[2], dog_value));
            }
        }
        None
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

        if det <= 0.0 {
            return true;
        }

        (trace * trace) / det > (CURVATURE_THRESHOLD + 1.0).powi(2) / CURVATURE_THRESHOLD
    }

    fn dynamic_image_to_ndarray(&self, img: &DynamicImage) -> Array2<f32> {
        match img {
            DynamicImage::ImageLuma8(buffer) => {
                // 从 ImageBuffer<Luma<u8>, _> 转换
                let (width, height) = buffer.dimensions();
                let array = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
                    buffer.get_pixel(x as u32, y as u32)[0] as f32
                });
                array
            }
            _ => {
                panic!("Unsupported image format");
            }
        }
    }

    /// 最近邻插值缩放
    fn resize_nearest_grayscale_simple(
        &self,
        image: &Array2<f32>,
        new_height: usize,
        new_width: usize,
    ) -> Array2<f32> {
        let (src_h, src_w) = image.dim();
        let scale_y = src_h as f32 / new_height as f32;
        let scale_x = src_w as f32 / new_width as f32;

        Array2::from_shape_fn((new_height, new_width), |(y, x)| {
            let src_y = (y as f32 * scale_y) as usize;
            let src_x = (x as f32 * scale_x) as usize;
            image[[src_y.min(src_h - 1), src_x.min(src_w - 1)]]
        })
    }
}


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as mio





def pixel_flow():
    file_path = 'dragon_mother.jpg'
    src_image1 = mio.imread(file_path)
    src_image2 = mio.imread(file_path)
    image1 = np.expand_dims(src_image1, axis=0)
    image2 = np.expand_dims(src_image2, axis=0)
    image = np.concatenate([image1, image2], axis=0)
    N, H, W, C = image.shape

    sess = tf.Session()

    def mesh_grid(height, width):
        y_line_space = tf.linspace(-1., 1., height)
        x_line_space = tf.linspace(-1., 1., width)
        x_coordinates, y_coordinates = tf.meshgrid(x_line_space, y_line_space)
        x_coordinates = tf.reshape(x_coordinates, shape=[-1])
        y_coordinates = tf.reshape(y_coordinates, shape=[-1])
        indices_grid = tf.stack([x_coordinates, y_coordinates], axis=0)
        indices_grid = tf.cast(indices_grid, tf.float32)
        return indices_grid

    def repeat(x, n_repeats):
        ones = tf.ones(shape=(1, n_repeats), dtype=tf.int32)
        x = tf.matmul(tf.reshape(x, (-1, 1)), ones)
        return tf.reshape(x, [-1])

    def zero_flow():
        v = tf.zeros(dtype=tf.float32, shape=(N, H, W, 1))
        h = tf.zeros(dtype=tf.float32, shape=(N, H, W, 1))
        flow = tf.concat([v, h], axis=3)
        return flow

    def panning_flow(vertical=0.1, horizontal=0.1):
        v = tf.constant(vertical, tf.float32, (N, H, W, 1))
        h = tf.constant(horizontal, tf.float32, (N, H, W, 1))
        flow = tf.concat([v, h], axis=3)
        return flow

    def rotate_flow(alpha, grid):
        x = tf.slice(grid, [0, 0], [1, -1])
        y = tf.slice(grid, [1, 0], [1, -1])
        new_x = x * tf.cos(alpha) - y * tf.sin(alpha)
        new_y = x * tf.sin(alpha) + y * tf.cos(alpha)
        return tf.concat([new_x, new_y], axis=0)

    def warp_flow(x, y, w, h, w_value, h_value):
        mask_v = np.zeros(dtype=np.float32, shape=(N, H, W, 1))
        mask_h = np.zeros(dtype=np.float32, shape=(N, H, W, 1))
        mask_v[:, y:y+h, x:x+w, :] = w_value
        mask_h[:, y:y+h, x:x+w, :] = h_value
        mask = np.concatenate((mask_v, mask_h), axis=3)
        flow = tf.constant(mask, dtype=tf.float32)
        return flow


    # input placeholder
    input = tf.placeholder(tf.float32, (N, H, W, C), 'input')

    # indices_grid limited in [-1, 1]
    indices_grid = mesh_grid(H, W)

    # some common pixel flow
    # flow is a matrix with size [N,2,H*W]
    flow = panning_flow(0.3, -0.2)
    flow = tf.reshape(tf.transpose(flow, [0, 3, 1, 2]), [-1, 2, H * W])
    transformed_grid = tf.add(flow, indices_grid)

    # rotate flow?
    # new_grid = rotate_flow(0.5, indices_grid)
    # new_grid = tf.expand_dims(new_grid, axis=0)
    # transformed_grid = tf.tile(new_grid, (N, 1, 1))

    # warp flow
    # flow = warp_flow(0, 0, 300, 300, 0.2, 0.5)
    # flow = tf.reshape(tf.transpose(flow, [0, 3, 1, 2]), [-1, 2, H * W])
    # transformed_grid = tf.add(flow, indices_grid)


    # flow[N,0,:] represents the ratio of vertical flow
    # flow[N,1,:] represents the ratio of horizontal flow
    x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
    x_s_flatten = tf.reshape(x_s, [-1])
    y_s_flatten = tf.reshape(y_s, [-1])

    # scale indices from [-1,1] --> [0,W-1] or [0,H-1]
    x = tf.cast((x_s_flatten + 1.0) * (W - 1) / 2.0, tf.int32)
    y = tf.cast((y_s_flatten + 1.0) * (H - 1) / 2.0, tf.int32)
    # clip the value which is not in [0, W-1] or [0, H-1]
    x = tf.clip_by_value(x, 0, W - 1)
    y = tf.clip_by_value(y, 0, H - 1)

    # get the indices
    flat_image_dims = H * W
    pixels_batch = tf.range(N) * flat_image_dims
    flat_output_dims = H * W
    base = repeat(pixels_batch, flat_output_dims)
    indices = base + y * W + x # store according to horizontal?

    # get the pixel value by the indices
    flat_image = tf.reshape(input, shape=(-1, C))
    flat_image = tf.cast(flat_image, dtype='float32')
    output = tf.gather(flat_image, indices)
    output = tf.reshape(output, (N,H,W,C))

    # plot the result
    out_image = sess.run(output, feed_dict={input: image})
    plt.subplot(121)
    plt.imshow(np.squeeze(image1[:, :, :].astype(np.uint8), axis=0))
    plt.subplot(122)
    dst_image = np.reshape(out_image[0, :, :, :].astype(np.uint8), (H, W, C))
    # mio.imsave('rotate+5_flow.jpg', dst_image)
    plt.imshow(dst_image)

    plt.show()





if __name__ == '__main__':
    pixel_flow()
import cv2
import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle


def vis_2d_points_img(show_image, joints):
    BoneVec_new = [
        (1, 2),
        (1, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (1, 17),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    BoneColorVec_point_21 = [
        (0, 0, 200),
        (0, 0, 200),
        (150, 0, 0),
        (200, 0, 0),
        (255, 0, 0),
        (100, 100, 0),
        (150, 150, 0),
        (200, 200, 0),
        (255, 255, 0),
        (0, 100, 50),
        (0, 150, 75),
        (0, 200, 100),
        (0, 255, 125),
        (0, 50, 100),
        (0, 75, 150),
        (0, 100, 200),
        (0, 125, 255),
        (120, 0, 120),
        (150, 0, 150),
        (200, 0, 200),
        (255, 0, 255),
    ]

    for jointIndex in range(len(BoneVec_new)):
        x1 = int(joints[BoneVec_new[jointIndex][0]][0])
        y1 = int(joints[BoneVec_new[jointIndex][0]][1])
        joint1 = (x1, y1)
        cv2.circle(show_image, joint1, 1, BoneColorVec_point_21[BoneVec_new[jointIndex][0]], -1)
        x2 = int(joints[BoneVec_new[jointIndex][1]][0])
        y2 = int(joints[BoneVec_new[jointIndex][1]][1])
        joint2 = (x2, y2)
        cv2.circle(show_image, joint2, 1, BoneColorVec_point_21[BoneVec_new[jointIndex][0]], -1)
        cv2.line(show_image, joint1, joint2, (0, 0, 0), 1)

    return show_image


def decode_keypoint(heatmap):
    kpys = []
    h, w = heatmap.shape[1:3]
    for i in range(21):
        hm = heatmap[i]
        idx = hm.argmax()
        x = idx % w
        y = idx // h
        kpys.append([x, y])
    return kpys


def draw_skeleton(image, kps, radius=1):
    # lines_index = [[0,13,14,15,16], [0,1,2,3,17], [0,4,5,6,18], [0,10,11,12,19], [0,7,8,9,20]]      # Interhand
    lines_index = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]  # predict monapth
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    ori_pt = kps[0]
    x, y = int(ori_pt[0]), int(ori_pt[1])
    cv2.circle(image, (x, y), radius=radius + 1, color=colors[0], thickness=-1)

    for n, l_idx in enumerate(lines_index):
        for i in range(1, 5):
            x, y = int(kps[l_idx[i]][0]), int(kps[l_idx[i]][1])
            cv2.circle(image, (x, y), radius=radius + 1, color=colors[n], thickness=-1)

            x0, y0 = int(kps[l_idx[i - 1]][0]), int(kps[l_idx[i - 1]][1])
            cv2.line(image, (x0, y0), (x, y), colors[n], radius)

    return image


def check_img_to_save(img_path, img, label, joint, joint2):
    img_path = img_path.replace("/", "#")
    # if label is not None:
    #     after_kpts = decode_keypoint(label)
    after_kpts = label
    test_image = (img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    test_image2 = test_image.copy()
    test_image3 = test_image.copy()    
    if label is not None:
        test_image = draw_skeleton(test_image, np.array(after_kpts) * 8)
        # test_image = draw_skeleton(test_image, np.array(label) * 8)
    test_image2 = draw_skeleton(test_image2, np.array(joint[:, :2]) * 8)
    test_image3 = draw_skeleton(test_image3, np.array(joint2[:, :2]) * 8)
    
    test_image = cv2.resize(test_image, (128, 128))
    test_image2 = cv2.resize(test_image2, (128, 128))
    test_image3 = cv2.resize(test_image3, (128, 128))
    saveimg = np.concatenate([test_image, test_image2, test_image3], axis=1)
    import os

    if not os.path.exists("outputs/check"):
        os.mkdir("outputs/check")
    cv2.imwrite("outputs/check/{}.jpg".format(img_path), saveimg)


def check_img_and_history_to_save(img_path, img, history_img, label, his_label):
    img_path = img_path.replace("/", "#")
    if label is not None:
        after_kpts = decode_keypoint(label)
    if his_label is not None:
        his_after_kpts = decode_keypoint(his_label)
    test_image = (img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    test_image2 = (history_img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
    test_image2 = cv2.cvtColor(test_image2, cv2.COLOR_GRAY2RGB)
    if label is not None:
        test_image = draw_skeleton(test_image, np.array(after_kpts) * 6)
        # test_image = draw_skeleton(test_image, np.array(label) * 8)
    if his_label is not None:
        test_image2 = draw_skeleton(test_image2, np.array(his_after_kpts) * 6)
    # test_image2 = draw_skeleton(test_image2, np.array(joint[:, :2]) * 8)

    test_image = cv2.resize(test_image, (128, 128))
    test_image2 = cv2.resize(test_image2, (128, 128))
    saveimg = np.concatenate([test_image, test_image2], axis=1)
    cv2.imwrite("outputs/check/{}.jpg".format(img_path), saveimg)


def eval_img_to_save(img_path, img, pred_joint, joint):
    img = img.cpu()
    joint = joint.cpu()
    pred_joint = pred_joint.cpu()
    test_image = (img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    test_image2 = test_image.copy()
    test_image = draw_skeleton(test_image, np.array(pred_joint[:, :2]) * 8)
    test_image2 = draw_skeleton(test_image2, np.array(joint[:, :2]) * 8)

    test_image = cv2.resize(test_image, (128, 128))
    test_image2 = cv2.resize(test_image2, (128, 128))
    saveimg = np.concatenate([test_image, test_image2], axis=1)
    cv2.imwrite("outputs/check_ori/{}.jpg".format(img_path), saveimg)


def check_dof_img_to_save(img_path, img, label, joint, pose, has_common_dof):
    import torch
    from manopth import demo
    from manopth.manolayer import ManoLayer
    from matplotlib import pyplot as plt

    img_path = img_path.replace("/", "#")

    if has_common_dof:
        mano_layer = ManoLayer(
            mano_root="/mnt/bn/wlxlmk/mlx/users/huhui.22/playground/manodatas/models",
            use_pca=False,
            flat_hand_mean=False,
        )

        pose = torch.tensor(pose.astype(np.float32))
        verts, joints = mano_layer.forward(pose.reshape(-1, 48), torch.zeros((1, 10)), torch.zeros((1, 3)))

        ax = None
        demo.display_hand(
            {"verts": verts.detach(), "joints": joints.detach()}, mano_faces=mano_layer.th_faces, ax=ax, show=False
        )
        plt.savefig("outputs/check/{}.jpg".format(img_path))
        # plt.cla()
        # plt.clf()
        plt.close()

    if label is not None:
        after_kpts = decode_keypoint(label)
    else:
        after_kpts = joint[:, :2]
    test_image = (img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    test_image2 = test_image.copy()
    test_image = draw_skeleton(test_image, np.array(after_kpts) * 8)
    test_image2 = draw_skeleton(test_image2, np.array(joint[:, :2]) * 8)

    test_image = cv2.resize(test_image, (128, 128))
    test_image2 = cv2.resize(test_image2, (128, 128))
    saveimg = np.concatenate([test_image, test_image2], axis=1)

    if has_common_dof:
        figimg = cv2.imread("outputs/check/{}.jpg".format(img_path))
        figimg = cv2.resize(figimg, (256, 256))
        saveimg = np.concatenate([saveimg, figimg], axis=0)

    cv2.imwrite("outputs/check/{}.jpg".format(img_path), saveimg)


def check_dof_img_and_history_to_save(img_path, img, prev_img, joint, prev_joint, pose, prev_pose, has_common_dof):
    import torch
    from manopth import demo
    from manopth.manolayer import ManoLayer
    from matplotlib import pyplot as plt

    img_path = img_path.replace("/", "#")

    if has_common_dof:
        mano_layer = ManoLayer(
            mano_root="extra_data/manodatas/models",
            use_pca=False,
            flat_hand_mean=False,
        )

        pose = torch.tensor(pose.astype(np.float32))
        verts, joints3d = mano_layer.forward(pose.reshape(-1, 48), torch.zeros((1, 10)), torch.zeros((1, 3)))

        ax = None
        demo.display_hand(
            {"verts": verts.detach(), "joints": joints3d.detach()}, mano_faces=mano_layer.th_faces, ax=ax, show=False
        )
        plt.savefig("outputs/check/{}.jpg".format(img_path))

        prev_pose = torch.tensor(prev_pose.astype(np.float32))
        verts, joints3d = mano_layer.forward(prev_pose.reshape(-1, 48), torch.zeros((1, 10)), torch.zeros((1, 3)))

        ax = None
        demo.display_hand(
            {"verts": verts.detach(), "joints": joints3d.detach()}, mano_faces=mano_layer.th_faces, ax=ax, show=False
        )
        plt.savefig("outputs/check/{}.jpg".format("prev_" + img_path))

    test_image = (img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    test_image = draw_skeleton(test_image, np.array(joint[:, :2]) * 8)
    test_image2 = (prev_img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
    test_image2 = cv2.cvtColor(test_image2, cv2.COLOR_GRAY2RGB)
    test_image2 = draw_skeleton(test_image2, np.array(prev_joint[:, :2]) * 8)

    test_image = cv2.resize(test_image, (128, 128))
    test_image2 = cv2.resize(test_image2, (128, 128))
    saveimg = np.concatenate([test_image, test_image2], axis=1)

    if has_common_dof:
        figimg = cv2.imread("outputs/check/{}.jpg".format(img_path))
        figimg = cv2.resize(figimg, (256, 256))
        prev_figimg = cv2.imread("outputs/check/{}.jpg".format("prev_" + img_path))
        prev_figimg = cv2.resize(prev_figimg, (256, 256))

        saveimg = np.concatenate([saveimg, figimg, prev_figimg], axis=0)

    cv2.imwrite("outputs/check/{}.jpg".format(img_path), saveimg)


def check_dof_img_and_history_to_save_multi_dataloader(transform_data_list):
    import torch
    from manopth import demo
    from manopth.manolayer import ManoLayer
    from matplotlib import pyplot as plt

    mano_layer = ManoLayer(
        mano_root="extra_data/manodatas/models",
        use_pca=False,
        flat_hand_mean=False,
    )

    img_path = ""
    # img_path = transform_data_list[0][0][1]["file_name"].replace("/", "_")
    big_imgs = []
    # big_pose_vis = []
    for samples_t in transform_data_list:
        img_views = []
        pose_vis_views = []
        for sample_view in samples_t:
            img, anno, targets = sample_view
            img = (img.add(0.5) * 255).cpu().data.numpy()[0].astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            joint2d = anno["joints_25d"].reshape(-1, 3)
            img = draw_skeleton(img, np.array(joint2d[:, :2]))
            img = cv2.resize(img, (128, 128))
            img_views.append(img)

            img_path += anno["file_name"].replace("/", "_")

            pose = anno["common_pose"]
            shape = anno["common_shape"]
            trans = anno["common_trans"]

            pose = torch.tensor(pose.astype(np.float32))
            shape = torch.tensor(shape.astype(np.float32))
            trans = torch.tensor(trans.astype(np.float32))
            verts, joints3d = mano_layer.forward(pose.reshape(-1, 48), shape.reshape(-1, 10), trans.reshape(-1, 3))

            Kvirt = anno["camI"]
            real_joint3d = anno["real_joint_3d"]
            joints2d_proj = np.matmul(real_joint3d, Kvirt.T)
            joints2d_proj /= joints2d_proj[:, 2:3]

            ax = None
            demo.display_hand(
                {"verts": verts.detach(), "joints": joints3d.detach()},
                mano_faces=mano_layer.th_faces,
                ax=ax,
                show=False,
            )
            plt.savefig("outputs/check/{}.jpg".format("tmp"))
            figimg = np.asarray(plt.gcf().canvas.buffer_rgba())[:, :, :3]
            plt.clf()
            # plt.savefig("outputs/check/{}.jpg".format(img_path))
            # figimg = cv2.imread("outputs/check/{}.jpg".format(img_path))
            figimg = cv2.resize(figimg, (256, 256))
            pose_vis_views.append(figimg)

        img_views = np.concatenate(img_views, axis=1)
        pose_vis_views = np.concatenate(pose_vis_views, axis=0)
        big_imgs.append(np.concatenate([img_views, pose_vis_views], axis=0))

    saveimg = np.concatenate(big_imgs, axis=0)
    tag = img_path.split("+")[0:2]
    tag = tag[0] + "+" + tag[1] + "+"
    cv2.imwrite("outputs/check/{}.jpg".format(tag + img_path.replace(tag, "").replace(".png", "")), saveimg)


def cal_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    inter_x1 = max(box1[0], box2[0])
    inter_x2 = min(box1[2], box2[2])
    inter_y1 = max(box1[1], box2[1])
    inter_y2 = min(box1[3], box2[3])
    if inter_y2 - inter_y1 <= 0 or inter_x2 - inter_x1 <= 0:
        return 0
    inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
    return inter_area / (area1 + area2 - inter_area)  # iou = inter / (area1 + area2 - inter)


def filename_hash(filename):
    file_name_split = filename.split("/")[-3:]
    file_name = ""
    for x in file_name_split:
        file_name = file_name + "/" + x
    import hashlib

    file_name_hash = int(hashlib.sha1(file_name.encode("utf-8")).hexdigest(), 16) % (2**32)
    return file_name_hash


def extend_bbox(bbox, image_shape=None):
    x1, y1, x2, y2 = bbox
    wb, hb = x2 - x1, y2 - y1
    scale = 0.15

    if image_shape is not None:
        height, width = image_shape
        newx1 = x1 - wb * scale if x1 - wb * scale > 0 else 0
        newx2 = x2 + wb * scale if x2 + wb * scale < width else width - 1
        newy1 = y1 - hb * scale if y1 - hb * scale > 0 else 0
        newy2 = y2 + hb * scale if y2 + hb * scale < height else height - 1
    else:
        newx1 = x1 - wb * scale
        newx2 = x2 + wb * scale
        newy1 = y1 - hb * scale
        newy2 = y2 + hb * scale
    exbox = [int(newx1), int(newy1), int(newx2), int(newy2)]
    return exbox


def box_ioa(rect1, rect2):
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2
    # s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    left = max(xmin2, xmin1)
    right = min(xmax2, xmax1)
    top = max(ymin2, ymin1)
    bottom = min(ymax2, ymax1)

    if left >= right or top >= bottom:
        return 0
    intersection = (right - left) * (bottom - top)
    return intersection / s2


def select_proposal(proposals, bbox, img_shape):
    candidate_box = list()
    candidate_box.append(extend_bbox(bbox, img_shape))
    for prop_conf in proposals:
        prop = prop_conf[:4]
        iou_pb = iou(prop, bbox)
        ioa_pb = box_ioa(prop, bbox)
        if iou_pb > 0.35 and ioa_pb > 0.8:
            candidate_box.append(prop)
    return candidate_box


def get_rectfy_rotation_angle(joint_2d):
    pt0 = joint_2d[0]
    pt1 = joint_2d[9]

    vec = pt1 - pt0
    angle = np.arctan2(vec[1], vec[0])

    if angle > 0:
        if angle - np.pi * 0.5 < 0:
            return 90, True
        else:
            return -90, False
    else:
        return 0, False


def rectfy_sample_by_rotate(img, joint_2d):
    height, width = img.shape[:2]

    angle, need_rot = get_rectfy_rotation_angle(joint_2d)

    if need_rot:
        if angle > 0:
            # roation 90 degree
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            joint_2d_new = joint_2d.copy()
            joint_2d_new[:, 0] = joint_2d[:, 1]
            joint_2d_new[:, 1] = width - joint_2d[:, 0]
            # joint_2d_new_last = joint_2d_last.copy()
            # joint_2d_new_last[:, 0] = joint_2d_last[:, 1]
            # joint_2d_new_last[:, 1] = width_last - joint_2d_last[:, 0]
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            joint_2d_new = joint_2d.copy()
            joint_2d_new[:, 0] = height - joint_2d[:, 1]
            joint_2d_new[:, 1] = joint_2d[:, 0]
            # joint_2d_new_last = joint_2d_last.copy()
            # joint_2d_new_last[:, 0] = height_last - joint_2d_last[:, 1]
            # joint_2d_new_last[:, 1] = joint_2d_last[:, 0]

        joint_2d = joint_2d_new
        # joint_2d_last = joint_2d_new_last
        # joint_2d_last = joint_2d
    return img, joint_2d


def get_rot6s(pose):
    num_pose = pose.shape[0]
    rot6s = list()
    for i in range(num_pose):
        rot6 = cv2.Rodrigues(pose[i])[0][:2]
        rot6s.append(rot6.reshape(-1))
    rot6s = np.array(rot6s)
    return rot6s


def matrix2dof(matrixs):
    # batch_size = matrixs.shape[0]
    # dofs = list()
    # for i in range(batch_size):
    #     dof = cv2.Rodrigues(matrixs[i].numpy())
    #     dofs.append(dof[0])
    # dofs = np.array(dofs)
    dofs = matrix_to_axis_angle(matrixs).reshape(-1, 3, 1)
    return dofs


def rot6tomatrix(rot6s):
    poses = rot6s.reshape(-1, 6)
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO

    # torch.det too slow, skip
    # assert (torch.stack([torch.det(mat) for mat in matrix]) < 0).sum() == 0

    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out


def rot6todof(rot6s):
    # rot6s = torch.from_numpy(rot6s)
    matrix = rot6tomatrix(rot6s)
    output_dofs = matrix2dof(matrix)
    output_dofs = output_dofs.reshape(-1, 3)

    return output_dofs * -1


def eular_to_mat(roll, pitch, yaw):
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    ry = np.asarray(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rz = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rx @ ry @ rz

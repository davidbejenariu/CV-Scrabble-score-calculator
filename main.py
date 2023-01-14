import copy

import cv2 as cv
import numpy as np
import os

from sklearn.svm import SVC
from skimage.feature import hog

from points import letter_score, letter_multiplier, word_multiplier


def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extract_square(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D((cx, cy), 0.15, 1.0)
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))

    cropped_image = rotated_image[1246:3370, 645:2595]
    width = 2025
    height = 2025

    puzzle = np.array([(0, 0), (1950, 0), (1950, 2124), (0, 2124)], dtype='float32')
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(cropped_image, M, (width, height))
    result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

    return result


def get_lines(size):
    vertical_lines = []
    horizontal_lines = []

    for i in range(0, size + 1, size // 15):
        vertical_lines.append([(i, 0), (i, size - 1)])
        horizontal_lines.append([(0, i), (size - 1, i)])

    return vertical_lines, horizontal_lines


def get_configuration(image, occupied, train=False, ans=None):
    matrix = np.empty((15, 15), dtype='str')
    vertical_lines, horizontal_lines = get_lines(2025)
    result = extract_square(image)
    _, thresh = cv.threshold(result, 160, 255, cv.THRESH_BINARY)

    patches = []
    labels = []

    if train:
        f = open(ans, 'r')
    else:
        test_patches = []

    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            y_min = vertical_lines[j][0][0] + 20
            y_max = vertical_lines[j + 1][1][0] - 20
            x_min = horizontal_lines[i][0][1] + 20
            x_max = horizontal_lines[i + 1][1][1] - 20

            patch = thresh[x_min:x_max, y_min:y_max].copy()
            patch_mean = np.mean(patch)

            if patch_mean > 150:
                if train:
                    patches.append(get_hog_descriptors(patch))

                if (i, j) not in occupied:
                    if train:
                        matrix[i][j] = f.readline()[-2]
                    else:
                        test_patches.append(get_hog_descriptors(patch))
                        matrix[i][j] = 'to be predicted'
                else:
                    matrix[i][j] = occupied[(i, j)]

                labels.append(matrix[i][j])
            else:
                matrix[i][j] = '#'

    if train:
        return matrix, patches, labels
    else:
        test_patches = np.array(test_patches)
        labels = model.predict(test_patches)

        return matrix, labels


def get_hog_descriptors(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    return hog(
        image,
        pixels_per_cell=(19, 19),
        orientations=8,
        cells_per_block=(5, 5),
        block_norm='L2-Hys',
        feature_vector=True
    )


def get_train_data(path, path_aux):
    files = os.listdir(path)
    train_patches = []
    train_labels = []

    for game in range(5):
        occupied = {}

        for turn in range(game * 40, (game + 1) * 40, 2):
            img = cv.imread(path + '/' + files[turn])
            matrix, patches, labels = get_configuration(img, occupied, train=True, ans=path + '/' + files[turn + 1])
            config = []

            for i in range(15):
                for j in range(15):
                    if matrix[i][j] != '#' and (i, j) not in occupied:
                        config.append(str(i + 1) + chr(65 + j))
                        occupied[(i, j)] = matrix[i][j]

            train_patches += patches
            train_labels += labels

    # auxiliary images
    for k in range(1, 3):
        occupied = {}
        img = cv.imread(f'{path_aux}/litere_{k}.jpg')
        matrix, patches, labels = get_configuration(img, occupied, train=True, ans=f'{path_aux}/litere_{k}.txt')
        config = []

        for i in range(15):
            for j in range(15):
                if matrix[i][j] != '#' and (i, j) not in occupied:
                    config.append(str(i + 1) + chr(65 + j))
                    occupied[(i, j)] = matrix[i][j]

        train_patches += patches
        train_labels += labels

    train_patches = np.array(train_patches)
    train_labels = np.array(train_labels)

    return train_patches, train_labels


def compute_score(board, words, l_multiplier, w_multiplier):
    total_score = 0
    new_words = {}

    # going right
    for i in range(15):
        j = 0

        while j < 15:
            # after finding a letter, we go to the right until there are no more letters
            if board[i][j] != '#':
                start_pos = (i, j)
                score = 0
                multiplier = 1

                while j < 15 and board[i][j] != '#':
                    multiplier = max(multiplier, w_multiplier[i][j])
                    score += letter_score[board[i][j]] * l_multiplier[i][j]
                    j += 1

                score *= multiplier
                end_pos = (i, j - 1)

                if (start_pos, end_pos) not in words and end_pos[1] - start_pos[1] > 0:
                    words[(start_pos, end_pos)] = score
                    new_words[(start_pos, end_pos)] = score
                    total_score += score
            else:
                j += 1

    # going down
    for j in range(15):
        i = 0

        while i < 15:
            # after finding a letter, we go to down until there are no more letters
            if board[i][j] != '#':
                start_pos = (i, j)
                score = 0
                multiplier = 1

                while i < 15 and board[i][j] != '#':
                    multiplier = max(multiplier, w_multiplier[i][j])
                    score += letter_score[board[i][j]] * l_multiplier[i][j]
                    i += 1

                score *= multiplier
                end_pos = (i - 1, j)

                if (start_pos, end_pos) not in words and end_pos[0] - start_pos[0] > 0:
                    words[(start_pos, end_pos)] = score
                    new_words[(start_pos, end_pos)] = score
                    total_score += score
            else:
                i += 1

    for start_pos, end_pos in new_words.keys():
        if start_pos[0] == end_pos[0]:
            for j in range(start_pos[1], end_pos[1] + 1):
                l_multiplier[start_pos[0]][j] = 1
                w_multiplier[start_pos[0]][j] = 1
        else:
            for i in range(start_pos[0], end_pos[0] + 1):
                l_multiplier[i][start_pos[1]] = 1
                w_multiplier[i][start_pos[1]] = 1

    return total_score, words, l_multiplier, w_multiplier


def generate_annotations(path):
    files = os.listdir(path)
    os.mkdir('352_Bejenariu_David_Cosmin')

    for game in range(5):
        occupied = {}
        words = {}
        l_multiplier = copy.deepcopy(letter_multiplier)
        w_multiplier = copy.deepcopy(word_multiplier)
        turn_count = 0

        for turn in range(game * 20, (game + 1) * 20, 1):
            turn_count += 1
            all_letters_bonus = 0

            if turn_count < 10:
                output = open(f'352_Bejenariu_David_Cosmin/{game + 1}_0{turn_count}.txt', 'x')
            else:
                output = open(f'352_Bejenariu_David_Cosmin/{game + 1}_{turn_count}.txt', 'x')

            img = cv.imread(path + '/' + files[turn])
            matrix, labels = get_configuration(img, occupied)
            config = []
            k = 0

            for i in range(15):
                for j in range(15):
                    if matrix[i][j] != '#' and (i, j) not in occupied:
                        matrix[i][j] = labels[k]
                        k += 1

                        config.append(str(i + 1) + chr(65 + j) + ' ' + matrix[i][j])
                        occupied[(i, j)] = matrix[i][j]

            if k == 7:
                all_letters_bonus = 50

            score, words, l_multiplier, w_multiplier = compute_score(matrix, words, l_multiplier, w_multiplier)
            config.append(score + all_letters_bonus)

            for line in config:
                output.write(str(line) + '\n')


x_train, y_train = get_train_data('antrenare', 'imagini_auxiliare')

model = SVC(C=10, kernel='linear')
model.fit(x_train, y_train)

generate_annotations('testare')

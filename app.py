import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import nltk
from nltk.corpus import words
import tensorflow as tf


nltk.download('words')
valid_words = set(words.words())


def is_gibberish(text, threshold=0.45):
	words_in_text = text.split()
	if not words_in_text:
		return True

	valid_count = sum(1 for word in words_in_text if word.lower() in valid_words)
	valid_ratio = valid_count / len(words_in_text)

	return valid_ratio < threshold


def rotate_image(image, angle):
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)

	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	new_w = int((h * sin) + (w * cos))
	new_h = int((h * cos) + (w * sin))

	M[0, 2] += (new_w / 2) - center[0]
	M[1, 2] += (new_h / 2) - center[1]

	rotated = cv2.warpAffine(image, M, (new_w, new_h))
	return rotated


def correct_orientation(image):
	h, w = image.shape[:2]
	new_image = image.copy()
	if h < w:
		new_image = rotate_image(image, 90)
	text = pytesseract.image_to_string(new_image)
	if is_gibberish(text):
		new_image = rotate_image(new_image, 180)
	return new_image


def resizeImg(img, scale=0.1):
	width = int(img.shape[1] * scale)
	height = int(img.shape[0] * scale)
	dimensions = (width, height)
	return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)


def compressImg(img):
	scale = round(900 / max(img.shape[:2]), 2)
	img_reshape = resizeImg(img, scale)
	img_gray = cv2.cvtColor(img_reshape, cv2.COLOR_BGR2GRAY)

	img_orn = correct_orientation(img_gray)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
	sharpened_image = cv2.filter2D(img_orn, -1, kernel)

	return sharpened_image


def save_tf_answers(img, qno=-1):
	try:
		img = compressImg(img)
		img = img[:, 3 * img.shape[1] // 5:]
		H, W = img.shape[0], img.shape[1]

		canny = cv2.Canny(img, 50, 200, None, 3)

		try:
			lines = cv2.HoughLines(canny, 1, np.pi / 180, 200)
			lines = np.reshape(lines, (lines.shape[0], 2))
			vertical_lines = []

			for i in lines:
				if abs(np.cos(i[1])) > 0.99:
					if np.cos(i[1]) < 0:
						i[0] = abs(i[0])
						i[1] = np.pi + i[1]
						vertical_lines.append(i)
					else:
						vertical_lines.append(i)

			vertical_lines = np.array(vertical_lines)
			vertical_lines = vertical_lines[vertical_lines[:, 0].argsort()]
			v_edges = [vertical_lines[0]]

			for i in range(1, len(vertical_lines)):
				if abs(vertical_lines[i][0] - v_edges[len(v_edges) - 1][0]) < W // 3:
					pass
				else:
					v_edges.append(vertical_lines[i])

			v_edges = np.array(v_edges, dtype=int)
			TFcol = img[:, v_edges[0][0]:v_edges[1][0]]
		except Exception as e:
			# print(e)
			return jsonify({"Error": "404"})

		cannyTF = cv2.Canny(TFcol, 10, 100, None, 3)
		boxTF = cannyTF.copy()
		horizontal = []
		kernel = 5
		tolerance = 10
		for i in range(kernel, cannyTF.shape[0] - kernel):
			h_line = False
			for j in range(tolerance, cannyTF.shape[1] - tolerance):
				for k in range(-1 * kernel, kernel):
					if cannyTF[i + k][j - 1] == 255 or cannyTF[i + k][j] == 255 or cannyTF[i + k][j + 1] == 255:
						h_line = True
						break
				else:
					h_line = False
					break
				h_line = True
			if h_line:
				horizontal.append(i)
				boxTF[i, :] = 0
			else:
				boxTF[i, :] = 255

		min_lines = []
		s, n = 0, 0

		for i in range(len(horizontal) - 1):
			if horizontal[i + 1] - horizontal[i] <= kernel:
				s += horizontal[i]
				n += 1
				if i == len(horizontal) - 2:
					min_lines.append(int(s / n))
			else:
				s += horizontal[i]
				n += 1
				min_lines.append(int(s / n))
				if i == len(horizontal) - 2:
					min_lines.append(horizontal[-1])
				s = 0
				n = 0

		try:
			n = len(min_lines)
			s = 0
			diff = []
			for i in range(0, n - 1):
				diff.append(min_lines[i + 1] - min_lines[i])
			diff = np.array(diff)
			mean_diff = np.mean(diff)
			std_diff = np.std(diff)
			ini_dist = 0

			if min_lines[0] < 10:
				min_lines = min_lines[1:]
			if qno < 0:
				pass
			else:
				try:
					min_lines = min_lines[:qno + 1]
				except:
					pass
		except Exception as e:
			return jsonify({"Error": "404"})

		splits = 0
		splitted_images = []
		img_height = 40
		img_width = 100
		for i in range(len(min_lines) - 1):
			sliced_img = TFcol[min_lines[i]:min_lines[i + 1], :]
			if sliced_img.shape[0] < 25 or sliced_img.shape[0] > 200:
				continue
			resized_img = cv2.resize(sliced_img, (img_width, img_height))
			_, binary_img = cv2.threshold(resized_img, 128, 255, cv2.THRESH_BINARY)
			rescaled_img = binary_img / 255.0
			splitted_images.append(rescaled_img)
			splits += 1
		splitted_images = np.array(splitted_images)
		return splitted_images
	except Exception as e:
		# print(e)
		return jsonify({"Error": "404"})

def evaluate_answers(model, image_path, correct_answers):
	img = cv2.imread(image_path)
	image = save_tf_answers(img, qno=-1)

	# Predict the output for each image using the model
	predictions = [model.predict(np.expand_dims(img, axis=0), verbose=0) for img in image]

	# Convert predictions to readable format
	predicted_answers = [np.argmax(pred) for pred in predictions]

	# Map numeric predictions to 'True', 'False', or 'Blank'
	prediction_map = {0: 'True', 1: 'False', 2: 'Empty'}
	predicted_answers = [prediction_map[pred] for pred in predicted_answers]

	# Calculate marks
	marks = []

	for pred, correct in zip(predicted_answers, correct_answers):
		if pred == correct:
			marks.append(1)
		else:
			marks.append(0)

	return marks


UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/predict', methods = ['POST'])
def upload_media():
	if 'file' not in request.files:
		return jsonify({'error': 'media not provided'}), 400
	file = request.files['file']
	if file.filename == '':
		return jsonify({'error': 'no file selected'}), 400

	file_path = ""
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

	directory = file_path
	model_path = "./model_mae_7.keras"
	model = tf.keras.models.load_model(model_path)
	correct_answers = ['False', 'False', 'False', 'False', 'False', 'False', 'True', 'True', 'True', 'True']

	marks = evaluate_answers(model, directory, correct_answers)

	return jsonify({'Marks': sum(marks)})


# if __name__ == '__main__':
# 	app.run(debug = True)

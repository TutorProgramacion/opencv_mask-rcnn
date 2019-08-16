#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

const float confThreshold = 0.50f;
const float maskThreshold = 0.40f;
const float nmsThreshold  = 0.30f;

vector<string> classes;

void post_process(cv::Mat& frame, const std::vector<Mat>& outs);
void load_classes_names();

void main() {

	// cargar el modelo pre-entrenado, debemos indicar la ruta de los archivos .pb y .pbtxt 
	dnn::Net net = dnn::readNet("data/files/mask_rcnn_inception_v2_coco_2018_01_28.pb",
								"data/files/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");

	// cargar la imagen de prueba
	Mat image = imread("data/files/image.jpg");

	// salir si la imagen no se ha podido cargar
	if (image.empty()) {
		cout << "Not image found." << endl;
		return;
	}
	
	Mat blob;

	// crear el blob 4D a partir de la imagen de entrada
	dnn::blobFromImage(image, blob, 1.0, Size(image.cols, image.rows), Scalar(), true, false);

	vector<String> outputNames = { "detection_out_final" , "detection_masks" };
	vector<Mat> outs;

	// establecer como entrada el blob creado anteriormente 
	net.setInput(blob);
	// correr la red neuronal para obtener las salidas para los nombres de la capas indicadas por outputNames
	net.forward(outs, outputNames);

	// leer el archivo que contiene los nombres de las classes
	load_classes_names();
	// procesar los resultados devueltos por la red y mostrarlos sobre la imagen de entrada
	post_process(image, outs);

	// visualizar los resultados
	imshow("Mask-RCNN", image);
	waitKey();
}

void post_process(cv::Mat& frame, const std::vector<Mat>& outs)
{
	if (frame.empty()) return;

	Mat boxes = outs[0];  // guarda las regiones rectangulares encontradas
	Mat masks = outs[1];  // guarda las mascaras correspondientes a las regiones 

	const int numDetections = boxes.size[2]; // cantidad de regiones 
	const int numClasses = masks.size[1];    // cantidad de mascaras

	const int frameW = frame.cols;
	const int frameH = frame.rows;

	std::vector<int>   classIds;    // ID de clase a la que pertenece el objeto
	std::vector<float> confidences; // puntaje 
	std::vector<Rect>  predBoxes;   // rectangulo

	for (int i = 0; i < numDetections; ++i) {

		// obtener datos de cada una de las regiones encontradas por la red, se organizan de la siguiente manera:
		// [batchId, classId, confidence, left, top, right, bottom] - 1x1xNx7
		float* box = (float*)boxes.ptr<float>(0, 0, i);
		float score = box[2];

		// nos quedamos con aquellas que superen el umbral establecido
		if (score > confThreshold) {

			int classId   = static_cast<int>(box[1]);
			int boxLeft   = static_cast<int>(frameW * box[3]);
			int boxTop    = static_cast<int>(frameH * box[4]);
			int boxRight  = static_cast<int>(frameW * box[5]);
			int boxBottom = static_cast<int>(frameH * box[6]);

			// convertimos los datos a uno de tipo cv::Rect 
			cv::Rect rect{ cv::Point{ boxLeft, boxTop }, cv::Point{ boxRight, boxBottom } };
			rect &= cv::Rect({ 0,0 }, frame.size());

			// guardamos los datos que nos interesen para su uso posterior
			classIds.emplace_back(classId);
			predBoxes.emplace_back(rect);
			confidences.emplace_back(score);
		}
	}

	std::vector<int> indices;
	// eliminar rectangulos o regiones redundantes
	cv::dnn::NMSBoxes(predBoxes, confidences, confThreshold, nmsThreshold, indices);

	for (size_t i = 0; i < indices.size(); ++i) {

		const int idx     = indices[i];
		const Rect box    = predBoxes[idx];
		const int classId = classIds[idx];
		const float conf  = confidences[idx];
  
		const Scalar color(255, 120, 147);
		const Scalar rect_color(243, 150, 33);

		// obtener la mascara definida para cada uno de los objetos detectados en la imagen de entrada
		Mat mask(masks.size[2], masks.size[3], CV_32F, masks.ptr<float>(static_cast<int>(i), classId));
		resize(mask, mask, box.size()); // redimensionar la mascara para que coincida con el tamano de la region
		mask = mask > maskThreshold;    // elimina aquellas partes de la mascara que no superen el umbral

		Mat coloredRoi;

		// resaltar en color diferente la mascara correspondiente al objeto detectado
		addWeighted(frame(box), 0.3, color, 0.7, 0, coloredRoi);
		coloredRoi.copyTo(frame(box), mask);

		// Dibujar rectangulo correspondiente a la region detectada
		rectangle(frame, box, rect_color);
		std::string label = format("%.2f", conf);

		// obtener el nombre de clase usando si ID devuelto por la red MASK-RCNN
		if (!classes.empty()) {
			label = classes[classId] + ": " + label + " %";
		}

		int baseLine = 0;

		// obtener las posiciones para dibujar el texto
		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
		cv::Rect label_rect{ box.tl() - cv::Point2i(0, baseLine + label_size.height), cv::Size(box.width, label_size.height + baseLine) };
		
		// dibujar rectangulo de fondo y texto informativo (nombre de clase + puntaje)
		rectangle(frame, label_rect, rect_color, cv::FILLED);
		putText(frame, label, box.tl() - cv::Point2i(-4, baseLine - 1), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(225), 1, LINE_AA);
	}
}


void load_classes_names() {

	cout << "loading class names..." << endl;

	// Load names of classes
	string classesFile = "data/files/object_detection_classes_coco.txt";
	ifstream ifs(classesFile.c_str());
	string line;

	// get and save names list
	while (getline(ifs, line)) classes.push_back(line);
}
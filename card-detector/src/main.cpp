//
// XIAO ESP32S3 Sense - Capture, segment, and classify 100x100 chunks
// Commands:
//   'r' or 'R' -> capture photo, segment into chunks, classify each
//

#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_heap_caps.h"
#include "model.h"


// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your model (generated from xxd)
//#include "NeuralNetwork.h"  // Your model header file

#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

// #include "model.cc"

// Model parameters
const int CHUNK_SIZE = 100;
const int NUM_CLASSES = 3;

// TFLite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  
  // Memory arena for TFLite (adjust size as needed)
  constexpr int kTensorArenaSize = 3 * 1024 * 1024;  // 3MB, tune based on your model
  uint8_t* tensor_arena = nullptr;
}

// Class names (adjust to match your training)
const char* CLASS_NAMES[] = {
  "low","mid","high"
};

// ---------------------------------------------------------
// Initialize TFLite Model
// ---------------------------------------------------------
bool initTFLite() {
  // Allocate tensor arena in PSRAM if available
  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena in PSRAM, trying DRAM...");
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    if (!tensor_arena) {
      Serial.println("Failed to allocate tensor arena!");
      return false;
    }
  }
  
  // Load model
  model = tflite::GetModel(g_card_id_model_data);  // From your .h file
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema version %d not supported. Supported version is %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  
  // Set up resolver with required ops
  static tflite::MicroMutableOpResolver<11> resolver;  // Adjust number based on your model
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddRelu();
  resolver.AddMean();
  // Add other ops your model needs
  
  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return false;
  }
  
  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Print input details
  Serial.println("Model loaded successfully!");
  Serial.printf("Input shape: [%d, %d, %d, %d]\n", 
                input->dims->data[0], input->dims->data[1], 
                input->dims->data[2], input->dims->data[3]);
  Serial.printf("Input type: %d\n", input->type);
  Serial.printf("Output shape: [%d, %d]\n", 
                output->dims->data[0], output->dims->data[1]);
  
  return true;
}

// ---------------------------------------------------------
// Initialize camera
// ---------------------------------------------------------
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  
  // Use grayscale for card recognition
  config.frame_size   = FRAMESIZE_UXGA;   // 1600 x 1200
  config.pixel_format = PIXFORMAT_GRAYSCALE;  // Changed to grayscale
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  Serial.println("Camera init success!");
  return true;
}

// ---------------------------------------------------------
// Extract 100x100 chunk from grayscale image
// ---------------------------------------------------------
void extractChunk(uint8_t* image, int img_width, int img_height,
                  int start_x, int start_y,
                  int8_t* chunk_buffer,
                  float in_scale, int in_zero_point) {
  // Extract and quantize to int8 according to model's input params
  for (int y = 0; y < CHUNK_SIZE; y++) {
    for (int x = 0; x < CHUNK_SIZE; x++) {
      int img_x = start_x + x;
      int img_y = start_y + y;
      int idx = y * CHUNK_SIZE + x;

      if (img_x >= img_width || img_y >= img_height) {
        // Treat out-of-bounds as black (0.0)
        float float_val = 0.0f;
        int32_t q = (int32_t)roundf(float_val / in_scale) + in_zero_point;
        q = std::max(-128, std::min(127, q));
        chunk_buffer[idx] = (int8_t)q;
        continue;
      }

      uint8_t pixel = image[img_y * img_width + img_x];

      // If you trained on [0,1]:
      float float_val = pixel / 255.0f;

      int32_t q = (int32_t)roundf(float_val / in_scale) + in_zero_point;
      q = std::max(-128, std::min(127, q));
      chunk_buffer[idx] = (int8_t)q;
    }
  }
}


// ---------------------------------------------------------
// Run inference on a chunk
// ---------------------------------------------------------
int classifyChunk(int8_t* chunk_buffer, float* confidence) {
  // Copy chunk to input tensor
  int8_t* input_data = input->data.int8;
  memcpy(input_data, chunk_buffer, CHUNK_SIZE * CHUNK_SIZE);
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    *confidence = 0.0f;
    return -1;
  }
  
  // Get output (dequantize if needed)
  int8_t* output_data = output->data.int8;
  float scale = output->params.scale;
  int zero_point = output->params.zero_point;
  
  // Find class with highest score
  int best_class = 0;
  float best_score = -1000.0f;
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    // Dequantize: float_value = (int8_value - zero_point) * scale
    float score = (output_data[i] - zero_point) * scale;
    if (score > best_score) {
      best_score = score;
      best_class = i;
    }
  }
  
  *confidence = best_score;
  return best_class;
}

// ---------------------------------------------------------
// Capture, segment, and classify
// ---------------------------------------------------------
void captureAndClassify() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  int img_width = fb->width;
  int img_height = fb->height;
  
  Serial.printf("Captured image: %dx%d\n", img_width, img_height);
  
  // Calculate number of chunks
  int num_chunks_x = (img_width + CHUNK_SIZE - 1) / CHUNK_SIZE;
  int num_chunks_y = (img_height + CHUNK_SIZE - 1) / CHUNK_SIZE;
  int total_chunks = num_chunks_x * num_chunks_y;
  
  Serial.printf("Segmenting into %d chunks (%dx%d grid)\n", 
                total_chunks, num_chunks_x, num_chunks_y);
  
  // Allocate buffer for one chunk
  int8_t* chunk_buffer = (int8_t*)malloc(CHUNK_SIZE * CHUNK_SIZE);
  if (!chunk_buffer) {
    Serial.println("Failed to allocate chunk buffer!");
    esp_camera_fb_return(fb);
    return;
  }
  
  // Process each chunk
  Serial.println("\n=== CLASSIFICATION RESULTS ===");
  Serial.println("Format: [chunk_index] (x,y) -> class (confidence)");
  Serial.println();
  
  int chunk_index = 0;
  for (int grid_y = 0; grid_y < num_chunks_y; grid_y++) {
    for (int grid_x = 0; grid_x < num_chunks_x; grid_x++) {
      int start_x = grid_x * CHUNK_SIZE;
      int start_y = grid_y * CHUNK_SIZE;
      
      float in_scale = input->params.scale;
      int   in_zp    = input->params.zero_point;

      extractChunk(fb->buf, img_width, img_height,
             start_x, start_y,
             chunk_buffer,
             in_scale, in_zp);
      // Classify
      float confidence;
      int predicted_class = classifyChunk(chunk_buffer, &confidence);
      
      // Print result
      if (predicted_class >= 0 && predicted_class < NUM_CLASSES) {
        Serial.printf("[%3d] (%4d,%4d) -> %3s (%.3f)\n", 
                      chunk_index, start_x, start_y,
                      CLASS_NAMES[predicted_class], confidence);
      } else {
        Serial.printf("[%3d] (%4d,%4d) -> ERROR\n", 
                      chunk_index, start_x, start_y);
      }
      
      chunk_index++;
    }
  }
  
  Serial.println("\n=== END RESULTS ===\n");
  
  // Cleanup
  free(chunk_buffer);
  esp_camera_fb_return(fb);
}

// ---------------------------------------------------------
// Arduino setup
// ---------------------------------------------------------
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println();
  Serial.println("XIAO ESP32S3 - Card Recognition with Chunking");
  Serial.println("Commands:");
  Serial.println("  r  -> capture, segment, and classify");
  Serial.println();

  if (!initCamera()) {
    Serial.println("Camera init failed. Halting.");
    while (true) delay(1000);
  }
  
  if (!initTFLite()) {
    Serial.println("TFLite init failed. Halting.");
    while (true) delay(1000);
  }

  Serial.println("Ready.");
}

// ---------------------------------------------------------
// Arduino loop
// ---------------------------------------------------------
void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'r' || c == 'R') {
      Serial.println("Capturing and classifying...\n");
      unsigned long start = millis();
      captureAndClassify();
      unsigned long elapsed = millis() - start;
      Serial.printf("Total processing time: %lu ms\n\n", elapsed);
    }
  }

  delay(10);
}
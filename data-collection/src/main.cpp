//
// XIAO ESP32S3 Sense - Capture JPEG and send over Serial
// Commands:
//   'r' or 'R' -> capture and send photo over serial
//

#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"         // disable brownout problems
#include "soc/rtc_cntl_reg.h"// disable brownout problems

#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM
#include "camera_pins.h"

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

  // JPEG so we can send directly
  config.frame_size   = FRAMESIZE_SVGA;   // 800x600; change if you want
  config.pixel_format = PIXFORMAT_JPEG;   // JPEG output
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;              // 0-63 (lower = better quality)
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
// Capture a photo and send over Serial
// ---------------------------------------------------------
bool captureAndSendOverSerial() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }

  if (fb->format != PIXFORMAT_JPEG) {
    Serial.println("Frame is not JPEG; cannot send directly.");
    esp_camera_fb_return(fb);
    return false;
  }

  // Simple framing:
  //   STARTIMG <size>\n
  //   <binary jpeg data>
  //   \nENDIMG\n
  //
  // Your PC-side script should:
  //   1) Read line "STARTIMG size"
  //   2) Read exactly <size> bytes as binary
  //   3) Then read until "ENDIMG"
  //
  uint32_t size = fb->len;
  Serial.printf("STARTIMG %u\n", (unsigned)size);

  // Send raw JPEG bytes
  Serial.write(fb->buf, fb->len);

  // Optional newline after binary, then end marker
  Serial.println();
  Serial.println("ENDIMG");
  Serial.printf("Sent %u bytes\n", (unsigned)size);

  esp_camera_fb_return(fb);
  return true;
}

// ---------------------------------------------------------
// Arduino setup
// ---------------------------------------------------------
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // disable brownout detector

  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println();
  Serial.println("XIAO ESP32S3 Sense - Capture & send JPEG over serial");
  Serial.println("Commands:");
  Serial.println("  r  -> capture photo and send over serial");
  Serial.println();

  if (!initCamera()) {
    Serial.println("Camera init failed. Halting.");
    while (true) {
      delay(1000);
    }
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
      Serial.println("Capturing photo...");
      if (captureAndSendOverSerial()) {
        Serial.println("Capture + send complete.\n");
      } else {
        Serial.println("Capture + send failed.\n");
      }
    }
  }

  delay(10);
}

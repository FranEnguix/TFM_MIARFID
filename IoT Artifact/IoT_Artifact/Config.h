#include <M5Core2.h>
#include "Templates.h"
#include "Base64.h"
#include <WiFi.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string>
#include "HTTPClient.h"
#include <Arduino_JSON.h>
#include "time.h"
#undef min

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_FC.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output_temp = nullptr;
TfLiteTensor* output_humi = nullptr;

constexpr int kTensorArenaSize = 1024*4;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

#define artifact_name "m5stack_core_2"
#define artifact_password "mirror"

// GLOBAL VARIABLES
char fiveserver_jid[40];

char* ssid_names[3] = {};
char* pass_names[3] = {};

char* xmpp_domains[4] = {};
char* xmpp_nodes[2] = {};

char* ssid     = ssid_names[0];
char* password = pass_names[0];

const char* ntpServer = "pool.ntp.org";
const char* weather_api_server = "http://192.168.67.7/weathermeasures/";
const long  gmtOffset_sec = 3600;
const int   daylightOffset_sec = 3600;
struct tm current_time;

const int xmpp_port = 5222;
char* xmpp_domain = xmpp_domains[0];
char* xmpp_node = xmpp_nodes[0];

boolean create_new_account = true;
boolean xmpp_verbose = false;

const int verbose_level = 10;


int value = 0;

boolean restore_comm = false;

int i = 0;
int btna = 0;

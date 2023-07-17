
#include "Config.h"

void send_create_artifact_message(void);
void connect_wifi_and_xmpp_server(void);
float getRandomNumber(float lower_bound, float upper_bound);
void configure_NTP_server(void);
void printLocalTime(void);
void update_current_time(void);
String httpGETRequest(const char* serverName);
void send_data_predict(void);

void vTaskPredictData(void *pvParameters);
void vTaskGenerateRandomNum(void *pvParameters);
void vTaskUpdateTime(void *pvParameters);
TaskHandle_t vTaskGenerateRandomNum_Handler;
TaskHandle_t vTaskUpdateTime_Handler;
TaskHandle_t vTaskPredictData_Handler;

// Use WiFiClient class to create TCP connections
WiFiClient client;

void setup() {
  Serial.begin(115200);

  // Wait for device connected.
  while (!Serial) {
    ;
  }

  connect_wifi_and_xmpp_server();
  delay(1000);
  configure_NTP_server();
  delay(1000);
  send_create_artifact_message();
  delay(1000);
  
  xTaskCreate(
    vTaskPredictData,
    "TaskPredictData",
    6144,  // Stack size
    NULL,
    2,
    &vTaskPredictData_Handler);

  xTaskCreate(
    vTaskUpdateTime,
    "TaskUpdateTime",
    2048,  // Stack size
    NULL,
    1,
    &vTaskUpdateTime_Handler);
  
}

void send_data_predict(void) {
  String sensorReadings = httpGETRequest(weather_api_server);
  JSONVar myObject = JSON.parse(sensorReadings);
  float input_data[72];

  if (JSON.typeof(myObject) == "undefined") {
    Serial.println("Parsing input failed!");
  } else {
    Serial.print("input data from server request = ");
    Serial.println(myObject);
    for (int i = 0; i < 72; i = i + 1) {
      double data = (double) myObject[i];
      input_data[i] = (float) data;
    }    
  }

  // TEMP init
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(_content_drive_MyDrive_projects_weather_2023_05_21_11_41_58_model_FC_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output_temp = interpreter->output(0);
  output_humi = interpreter->output(1);
  

  int8_t x_quantized[72];
 

  for (byte i = 0; i < 72; i = i + 1) {
        input->data.int8[i] = input_data[i] / input->params.scale + input->params.zero_point; // convert to int8
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // Dequantize the output from integer to floating-point
  int8_t y_humi_q = output_humi->data.int8[0];
  float y_humi = (y_humi_q - output_humi->params.zero_point) * output_humi->params.scale;  
  Serial.print("Humidity: ");
  Serial.print(y_humi);
  Serial.print("\n");
  int8_t y_temp_q = output_temp->data.int8[0];
  float y_temp = (y_temp_q - output_temp->params.zero_point) * output_temp->params.scale;
  Serial.print("Temperature: ");
  Serial.print(y_temp);
  Serial.print("\n");

  float humidity = y_humi * 100;
  float temperature = y_temp * 60;
  char msg[70];
    snprintf(msg, 70, "{\"commandName\":\"displayData\",\"data\":[\"%.2f°C\",\"%.1f%%\"]}", temperature, humidity);
    if (verbose_level > 4) {
      Serial.println(msg);
    }
    send_msg(fiveserver_jid, msg);
}

void connect_wifi_and_xmpp_server(void) {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  // XMPP
  doConnection();
  delay(5000);
  init_communication_with_server();
  delay(500);
}

void configure_NTP_server(void) {
  // Init and get the time
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  if (verbose_level > 0) {
    update_current_time();
    printLocalTime();
  }
}

void send_create_artifact_message(void) {
  char temp_buffer_0[30];

  

  char *message2send = "{\"commandName\":\"createArtifact\",\"data\":[\"M5stack\",\"THArtifact\",\"{\\\"x\\\":-1.5,\\\"y\\\":6,\\\"z\\\":15}\"]}";
  char msg[600];

  if (verbose_level > 4) {
    Serial.println(xmpp_node);
    Serial.println(xmpp_domain);
  }
  strcpy(temp_buffer_0, xmpp_node);
  strcat(temp_buffer_0, "@");
  strcpy(fiveserver_jid, temp_buffer_0);
  strcat(fiveserver_jid, xmpp_domain);


  strcpy(msg, message2send);


  Serial.println(fiveserver_jid);
  Serial.println(msg);


  send_msg(fiveserver_jid, msg);
  delay(1000);
  xmpp_process_input();
  delay(1000);

}

void loop() {
  sleep(100);
}

float getRandomNumber(float lower_bound, float upper_bound) {
  float range = upper_bound - lower_bound;
  float random_float = (float)random(10000) / 10000.0;
  random_float = (random_float * range) + lower_bound;
  return random_float;
}

void vTaskGenerateRandomNum(void *pvParameters) {
  int msDelayTask1 = 3000;
  const TickType_t delay = pdMS_TO_TICKS(msDelayTask1);
  for (;;) {
    float temperature = getRandomNumber(20.0, 25.0);
    float humidity = getRandomNumber(40.0, 50.0);


    char msg[70];

    snprintf(msg, 70, "{\"commandName\":\"displayData\",\"data\":[\"%.2f°C\",\"%.1f%%\"]}", temperature, humidity);
    if (verbose_level > 4) {
      Serial.println(msg);
    }
    send_msg(fiveserver_jid, msg);
    vTaskDelay(delay);
  }
}

void vTaskPredictData(void *pvParameters) {
  int msDelayTask1 = 60000;
  const TickType_t delay = pdMS_TO_TICKS(msDelayTask1);
  for (;;) {
    send_data_predict();
    vTaskDelay(delay);
  }
}

void vTaskUpdateTime(void *pvParameters) {
  int msDelayTask = 60 * 1000;
  const TickType_t delay = pdMS_TO_TICKS(msDelayTask);
  for (;;) {
    update_current_time();
    if (verbose_level > 3) {
      printLocalTime();
    }
    vTaskDelay(delay);
  }
}

void update_current_time(void) {
  bool success = false;
  while (!success) {
    success = getLocalTime(&current_time);
    Serial.println("Failed to connect NTP server. Trying again...");
  }
}

void printLocalTime(void) {
  update_current_time();
  Serial.println(&current_time, "%A, %B %d %Y %H:%M:%S");
  Serial.print("Day of week: ");
  Serial.println(&current_time, "%A");
  Serial.print("Month: ");
  Serial.println(&current_time, "%B");
  Serial.print("Day of Month: ");
  Serial.println(&current_time, "%d");
  Serial.print("Year: ");
  Serial.println(&current_time, "%Y");
  Serial.print("Hour: ");
  Serial.println(&current_time, "%H");
  Serial.print("Hour (12 hour format): ");
  Serial.println(&current_time, "%I");
  Serial.print("Minute: ");
  Serial.println(&current_time, "%M");
  Serial.print("Second: ");
  Serial.println(&current_time, "%S");

  Serial.println("Time variables");
  char timeHour[3];
  strftime(timeHour,3, "%H", &current_time);
  Serial.println(timeHour);
  char timeWeekDay[10];
  strftime(timeWeekDay,10, "%A", &current_time);
  Serial.println(timeWeekDay);
  Serial.println();
}

String httpGETRequest(const char* serverName) {
  HTTPClient http;

  // Your IP address with path or Domain name with URL path 
  http.begin(serverName);


  // Send HTTP POST request
  int httpResponseCode = http.GET();

  String payload = "{}"; 

  if (httpResponseCode>0) {
    Serial.print("HTTP Response code: ");
    Serial.println(httpResponseCode);
    payload = http.getString();
  }
  else {
    Serial.print("Error code: ");
    Serial.println(httpResponseCode);
  }
  // Free resources
  http.end();

  return payload;
}

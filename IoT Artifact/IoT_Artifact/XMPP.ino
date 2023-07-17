
void doConnection() {

  Serial.print("Connecting to XMPP Server...");
  Serial.println(xmpp_domain);

  if (!client.connect(xmpp_domain, xmpp_port)) {
    Serial.println("Connection to XMPP Server Failed...");
    M5.Lcd.fillScreen(RED);
    return;
  } else {
    Serial.println("Connection to XMPP Server Accepted...");
  }
}


void init_communication_with_server() {
  sendTemplate(open_stream_template, strlen(xmpp_domain), xmpp_domain);
  delay(300);
  //client.flush();

  if (create_new_account == true) {
    sendTemplate(ask_to_create_new_acount, 0);
    delay(300);
    if (xmpp_verbose == true) {
      read_server_1();
      Serial.println("----------------------------------------------------------------");
    }
    sendTemplate(create_new_acount, strlen(artifact_name) + strlen(artifact_password), artifact_name, artifact_password);
    delay(300);
    if (xmpp_verbose == true) {
      read_server_1();
      Serial.println("----------------------------------------------------------------");
    }
    create_new_account = false;
  }

  doAutentication(artifact_name, artifact_password);
  delay(300);
  //client.flush();
  sendTemplate(open_stream_template, strlen(xmpp_domain), xmpp_domain);
  delay(300);
  //client.flush();
  sendTemplate(bind_template, strlen(xmpp_domain), xmpp_domain);
  delay(300);
  //client.flush();
  sendTemplate(session_request_template, strlen(xmpp_domain), xmpp_domain);
  delay(300);
  //client.flush();
}

//---------------------------------------------------------------------------------------------------------------------
// Presence
//---------------------------------------------------------------------------------------------------------------------

void unsubscribe_presence(char *jid_to, char * jid_from, boolean bverbose) {
  //client.flush();
  // Unsubcribe presence
  sendTemplate(presence_unsubscribe_template, strlen(jid_to) + strlen(jid_from), jid_from, jid_to);
  delay(300);

  if (bverbose) {
    read_server_1();
    Serial.println("----------------------------------------------------------------");
  }
}

void subscribe_presence(char *jid_to, char * jid_from, boolean bverbose) {
  //client.flush();
  presence_show(true);

  // Subcribe presence
  sendTemplate(presence_subscribe_template, strlen(jid_to) + strlen(jid_from), jid_from, jid_to);
  delay(300);

  if (bverbose) {
    read_server_1();
    Serial.println("----------------------------------------------------------------");
  }
}


void presence_online(char *jid_to, char * jid_from, boolean bverbose) {
  //client.flush();
  // Unsubcribe presence
  sendTemplate(presence_online_template, strlen(jid_to) + strlen(jid_from), jid_to, jid_from);
  delay(300);

  if (bverbose) {
    read_server_1();
    Serial.println("----------------------------------------------------------------");
  }
}

void presence_show(boolean bverbose) {
  //client.flush();
  // Unsubcribe presence
  sendTemplate(show_presence_template, 0);
  delay(300);

  if (bverbose) {
    read_server_1();
    Serial.println("----------------------------------------------------------------");
  }
}

void presence_status(char *jid_to, char * type_status, boolean bverbose) {
  sendTemplate(presence_status_template, strlen(jid_to) + strlen(type_status), jid_to, type_status);
  delay(300);

  if (bverbose) {
    read_server_1();
    Serial.println("----------------------------------------------------------------");
  }
}

//---------------------------------------------------------------------------------------------------------------------
// Message
//---------------------------------------------------------------------------------------------------------------------

void send_msg(char *RX_Jid, char *message2send) {
  sendTemplate(message_template, strlen(RX_Jid) + strlen(message2send), RX_Jid, message2send);
  delay(300);
}

void read_server() {
  // Read all the lines of the reply from server and print them to Serial
  Serial.print("Client Available: ");
  Serial.println(client.available());
  for (int i = 0; i < 200; i++) {
    while (client.available()) {
      String client_rx = client.readString();
      Serial.print(client_rx);
      
    }
    delay(10);
  }
  Serial.println();
}

void read_server_1() {
  while (client.available()) {
    char c = client.read();
    Serial.print(c);
  }
  Serial.println();
}

void doAutentication(char *username, char *password) {
  int plainStringLen = strlen(username) + strlen(password) + 2;
  int encStringLen = base64_enc_len(plainStringLen);
  char plainString[plainStringLen];
  char encString[encStringLen];

  /* Set up our plain auth string. It's in the form:
     "\0username\0password"
     where \0 is the null character
  */
  memset(plainString, '\0', plainStringLen);
  memcpy(plainString + 1, username, strlen(username));
  memcpy(plainString + 2 + strlen(username), password, strlen(password));

  /* Encode to base64 */
  base64_encode(encString, plainString, plainStringLen);
  sendTemplate(plain_auth_template, encStringLen, encString);
}

void doEncoding(char *data2encod) {
  int plainStringLen = strlen(data2encod) + 2;
  int encStringLen = base64_enc_len(plainStringLen);
  char plainString[plainStringLen];
  char encString[encStringLen];

  /* Set up our plain auth string. It's in the form:
     "\0username\0password"
     where \0 is the null character
  */
  memset(plainString, '\0', plainStringLen);
  memcpy(plainString + 1, data2encod, strlen(data2encod));

  /* Encode to base64 */
  base64_encode(encString, plainString, plainStringLen);

  Serial.println(encString);
}

//-------------------------------------------------
// Send Templates
//-------------------------------------------------

int sendTemplate(const prog_char *temp_P, int fillLen, ...) {
  int tempLen = strlen_P(temp_P);
  char temp[tempLen];
  char buffer[tempLen + fillLen];
  va_list args;

  strcpy_P(temp, temp_P);

  va_start(args, fillLen);
  vsprintf(buffer, temp, args);

  if (xmpp_verbose == true) {
    Serial.println(buffer);
  }
  client.write(buffer);
  return 1;
}

void xmpp_process_input() {
  int i = 0;
  int bufLen = 8;
  int init_stimate_body = 0;
  int end_stimate_body = 0;
  int init_stimate_from = 0;
  int end_stimate_from = 0;

  String body_msg = " ";
  String from_msg = " ";

  char *data;
  char buffer[bufLen];
  memset(buffer, '\0', bufLen);
  boolean stateChanged = false;

  if (client.available()) {
    String client_rx = client.readString();

    init_stimate_body = client_rx.indexOf("<body>");
    end_stimate_body = client_rx.indexOf("</body>");
    body_msg = client_rx.substring(init_stimate_body + strlen("<body>"), end_stimate_body);

    init_stimate_from = client_rx.indexOf("from='");
    end_stimate_from = client_rx.indexOf("' type");
    from_msg = client_rx.substring(init_stimate_from + strlen("from='"), end_stimate_from);

    
    if (sizeof(body_msg)) {
      M5.Lcd.setRotation(1);
      M5.Lcd.fillScreen(BLACK);
      M5.Lcd.setCursor(0, 10);
      M5.Lcd.setTextColor(WHITE);
      M5.Lcd.setTextSize(3);
      M5.Lcd.print(body_msg);
    }
    //lostConnection();

    //initSubscription();
    //unSubscription();

    ////client.flush();
    M5.update();
  }
  if (WiFi.status() != WL_CONNECTED) {
    M5.Lcd.setRotation(3);
    M5.Lcd.fillScreen(RED);
    M5.Lcd.setCursor(0, 10);
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.setTextSize(2);
    M5.Lcd.println("Lost Connection, start connecting to: ");
    M5.Lcd.println(ssid);

    WiFi.begin(ssid, password);
    int j = 0;
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      /*M5.Lcd.setCursor(j, 20);
      M5.Lcd.setTextColor(WHITE);
      M5.Lcd.setTextSize(3);
      M5.Lcd.print(".");
      j = j + 10;
      if (j > 100) {
        j = 0;
        M5.Lcd.print("Lost Connection...");
        M5.Lcd.fillScreen(RED);
      }*/
    }
    doConnection();
    init_communication_with_server();
    presence_show(true);
    M5.Lcd.fillScreen(BLUE);
  }

}

//----------------------------------------------------------------------
// Subscription
//----------------------------------------------------------------------

void initSubscription() {
  if (M5.BtnA.isPressed()) {
    delay(50);
    M5.update();
    //M5.Beep.tone(4000);

    M5.Lcd.setRotation(3);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setCursor(0, 10);
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.setTextSize(3);
    M5.Lcd.print("Init Subscription...");
    subscribe_presence("receiver1@192.168.1.20", "mbed@192.168.1.20", true);
    //M5.Beep.mute();
    M5.Lcd.fillScreen(BLACK);
    delay(2000);
  }
}

void unSubscription() {
  if (M5.BtnB.isPressed()) {
    delay(50);
    M5.update();
    //M5.Beep.tone(4000);

    M5.Lcd.setRotation(3);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setCursor(0, 10);
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.setTextSize(3);
    M5.Lcd.print("Unsubscription...");
    unsubscribe_presence("receiver1@192.168.1.20", "mbed@192.168.1.20", true);
    //M5.Beep.mute();
    M5.Lcd.fillScreen(BLACK);
    delay(2000);
  }
}

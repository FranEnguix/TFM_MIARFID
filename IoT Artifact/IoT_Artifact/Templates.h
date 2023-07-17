static const prog_char PROGMEM open_stream_template[] = "<stream:stream "
    "xmlns='jabber:client' "
    "xmlns:stream='http://etherx.jabber.org/streams' "
    "to='%s' "
    "version='1.0'>";

static const char PROGMEM ask_to_create_new_acount[] = "<iq type='get' "
    "id='reg1'> "
    "<query xmlns='jabber:iq:register'/>"
    "</iq>";

static const char PROGMEM create_new_acount[] = "<iq type='set' "
    "id='reg2'> "
    "<query xmlns='jabber:iq:register'> "
    "<username>%s</username> "
    "<password>%s</password> "
    "</query> </iq>";

static const prog_char PROGMEM plain_auth_template[] = "<auth "
    "xmlns='urn:ietf:params:xml:ns:xmpp-sasl' "
    "mechanism='PLAIN'>"
    "%s"
    "</auth>";

static const prog_char PROGMEM bind_template[] = "<iq "
    "type='set' "
    "id='bind_1'>"
    "<bind "
    "xmlns='urn:ietf:params:xml:ns:xmpp-bind'>"
    "<resource>%s</resource>"
    "</bind>"
    "</iq>";

static const prog_char PROGMEM session_request_template[] = "<iq "
    "to='%s' "
    "type='set' "
    "id='ard_sess'>"
    "<session "
    "xmlns='urn:ietf:params:xml:ns:xmpp-session' />"
    "</iq>";

static const prog_char PROGMEM message_template[] = "<message "
    "to='%s' "
    "xmlns='jabber:client' "
    "type='chat' "
    "id='msg' "
    "xml:lang='en'>"
    "<body>%s</body>"
    "</message>";

static const prog_char PROGMEM close_template[] = "<presence type='unavailable'/> "
    "</stream:stream>";


//---------------------------------------------------------------------------------------------------------------------
// Templates to Presence
//---------------------------------------------------------------------------------------------------------------------

static const prog_char PROGMEM show_presence_template[] = "<presence> "
    "<show/>"
    "</presence>";

static const prog_char PROGMEM status_presence_template[] = "<presence>"
    "<status/> element MUST NOT possess any attributes<status/>"
    "</presence>";

static const prog_char PROGMEM specifying_presence_priority_template[] = "<presence xml:lang='en'>"
    "<show>dnd</show>"
    "<status>Wooing Juliet</status>"
    "<status xml:lang='en'>Ja dvo&#x0159;&#x00ED;m mbed</status>"
    "<priority>1</priority>"
    "</presence>";

static const prog_char PROGMEM presence_subscribe_template[] = "<presence "
    "from='%s' "
    "id='xk3h1v69' "
    "to='%s' "
    "type='subscribe'/> ";

static const prog_char PROGMEM presence_unsubscribe_template[] = "<presence "
    "from='%s' "
    "to='%s' "
    "type='unsubscribe'/>";

static const char PROGMEM presence_online_template[] = "<presence "
    "from='%s' "
    "to='%s' "
    "type='online'/>";

static const char PROGMEM presence_status_template[] = "<presence "
    "to='%s' "
    "<show>away</show> "
    "<status>'%s'</status>";

//---------------------------------------------------------------------------------------------------------------------
// Pub-Sub
//---------------------------------------------------------------------------------------------------------------------


static const char PROGMEM pub_sub_template[] = "<iq type='set' "
    "from='mbed@192.168.1.20' "
    "to='receiver1@192.168.1.20' "
    "id='pub1'> "
    "<pubsub xmlns='http://jabber.org/protocol/pubsub'> "
    "<publish node='embed_node'> "
    "<item> "
    "<entry xmlns='http://www.w3.org/2005/Atom'> "
    "<title>Test</title> "
    "<summary> "
    "Hola mi primer Pub-sub"
    "</summary> "
    "<link rel='alternate' type='text/html' "
    "href='http://denmark.lit/2003/12/13/atom03'/> "
    "<id>tag:denmark.lit,2003:entry-32397</id> "
    "<published>2003-12-13T18:30:02Z</published> "
    "<updated>2003-12-13T18:30:02Z</updated> "
    "</entry> "
    "</item> "
    "</publish> "
    "</pubsub> "
    "</iq>";


// Init XMPP
/*static const char PROGMEM open_stream_template[] = "<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' to='192.168.0.110' version='1.0'>";
  static const char PROGMEM plain_auth_template[] = "<auth xmlns='urn:ietf:params:xml:ns:xmpp-sasl' mechanism='PLAIN'>%s</auth>";

  static const char PROGMEM plain_auth_template_1[] = "<auth xmlns='urn:ietf:params:xml:ns:xmpp-sasl' mechanism='PLAIN'>AG01ZXBkX2VzcF8zMgA=</auth>";

  static const char PROGMEM ask_to_create_new_acount[] = "<iq type='get' id='reg1'> <query xmlns='jabber:iq:register'/> </iq>";
  static const char PROGMEM create_new_acount[] = "<iq type='set' id='reg2'> <query xmlns='jabber:iq:register'> <username>mbed</username> <password>mirror</password> </query> </iq>";

  // Session Sstablishment
  static const char PROGMEM session_establishment[] = "<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' id='c2s_345' from='192.168.0.110' version='1.0'> <stream:features> <bind xmlns='urn:ietf:params:xml:ns:xmpp-bind'/> <session xmlns='urn:ietf:params:xml:ns:xmpp-session'/> </stream:features>";

  // Template MSG
  static const char PROGMEM init_message_template[] = "<message from='mbed@192.168.0.110' to='receiver1@192.168.0.110' type='chat'> <composing xmlns='http://jabber.org/protocol/chatstates'/> </message>";

  static const char PROGMEM message_template[] = "<message to='receiver1@192.168.0.110' xmlns='jabber:client' type='chat' id='msg' xml:lang='en'> <body>Hola Soy un mbed</body> </message>";
  static const char PROGMEM message_template_1[] = "<message type='chat' to='receiver1@192.168.0.110' id='aae9a'> <body>asd</body> <active xmlns='http://jabber.org/protocol/chatstates'/> </message>";
  static const char PROGMEM message_template_2[] = "<message type='chat' to='receiver1@192.168.0.110' id='aae9a'> <body>asd</body> <active xmlns='http://jabber.org/protocol/chatstates'/> </message>";
  static const char PROGMEM message_template_3[] = "<message to='receiver1@192.168.0.110' from='mbed@192.168.0.110' type='chat'> <body>Art thou not Romeo, and a Montague?</body> <active xmlns='http://jabber.org/protocol/chatstates'/> </message>";
  static const char PROGMEM message_template_4[] = "<message to='receiver1@192.168.0.110' from='mbed@192.168.0.110' type='chat' xml:lang='en'> <body>Wherefore art thou, Romeo?</body> </message>";
  static const char PROGMEM message_template_5[] = "<message from='mbed@192.168.0.110/esp_32' to='receiver1@192.168.0.110' type='chat'> <body>Hi honey!</body> <active xmlns='http://jabber.org/protocol/chatstates'/> </message>";
  static const char PROGMEM message_template_6[] = "<message to='mbed@192.168.0.110' xmlns='jabber:client' type='chat' id='msg' xml:lang='en'> <body>Hola HP...</body> </message>";
  static const char PROGMEM message_template_7[] = "<message from='mbed@192.168.0.110' to='receiver1@192.168.0.110' type='chat'> <body>Hi honey!</body>  <active xmlns='http://jabber.org/protocol/chatstates'/> </message>";
  static const char PROGMEM message_template_8[] = "<message from='mbed@192.168.0.110' to='receiver1@192.168.0.110' xml:lang='en'> <body>Art thou not Romeo, and a Montague?</body> </message>";
  //static const char PROGMEM message_template_9[] = "<message from='mbed@192.168.0.110' id='mbed-4.1.247' to='receiver1@192.168.0.110'> <body>My lord, dispatch; read o'er these articles.</body> <request xmlns='urn:xmpp:receipts'/> </message>";


  // Presence Templates
  //static const char PROGMEM presence_template[] = "<presence from='mbed@192.168.0.110' to='receiver1@192.168.0.110' xml:lang='en'> <show>away</show> <status>I shall return!</status> <priority>1</priority> </presence>";
  static const char PROGMEM presence_template[] = "<presence xmlns='jabber:client' from='mbed@192.168.0.110' to='agg_1@192.168.0.110'><show>chat</show><status>available</status></presence>";
  static const char PROGMEM presence_template_1[] = "<presence from='mbed@192.168.0.110' to='receiver1@192.168.0.110' type='online'/>";
  static const char PROGMEM presence_template_2[] = "<presence to='receiver1@192.168.0.110'> <show>away</show> <status>Having a spot of tea</status> </presence>";
  static const char PROGMEM presence_template_3[] = "<presence><show/></presence>";
  static const char PROGMEM presence_template_4[] = "<presence from='mbed@192.168.0.110' to='receiver1@192.168.0.110' type='probe' />";


  static const char PROGMEM user_nickname_3[] = "<presence from='mbed@192.168.0.110' to='receiver1@192.168.0.110' type='subscribe'> <nick xmlns='http://jabber.org/protocol/nick'>M5</nick> </presence>";
  static const char PROGMEM initial_service_discovery_information_request[] = "<iq from='mbed@192.168.0.110' id='disco1' to='receiver1@192.168.0.110' type='get'> <query xmlns='http://jabber.org/protocol/disco#info'/> </iq>";

  // TLS
  static const char PROGMEM start_tls[] = "<starttls xmlns='urn:ietf:params:xml:ns:xmpp-tls'/>";
  static const char PROGMEM start_tls_1[] = "<stream:features> <starttls xmlns='urn:ietf:params:xml:ns:xmpp-tls'> <required/> </starttls> <mechanisms xmlns='urn:ietf:params:xml:ns:xmpp-sasl'> <mechanism>PLAIN</mechanism></mechanisms></stream:features>";
  static const char PROGMEM clients_request_a_new_session[] = "<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams&#8217' to='192.168.0.110' version='1.0'>";
*/


/*int sendTemplate(char *temp_P ) {

  int tempLen = strlen_P(temp_P);
  char temp[tempLen];

  strcpy_P(temp, temp_P);

  Serial.println(temp);
  client.write(temp);
  client.write('\n');

  return 1;
  }

  int sendTemplate_1(const char *temp_P, int fillLen, char *m1 ) {
  int tempLen = strlen_P(temp_P);
  char temp[tempLen];
  char buffer[tempLen + fillLen];
  strcpy_P(temp, temp_P);
  sprintf(buffer, temp, m1);
  Serial.println(buffer);
  client.write(buffer);
  client.write('\n');


  return 1;
  }

  int sendTemplate_2(const char *temp_P, int fillLen, char *m1, char *m2 ) {
  int tempLen = strlen_P(temp_P);
  char temp[tempLen];
  char buffer[tempLen + fillLen];

  strcpy_P(temp, temp_P);

  sprintf(buffer, temp, m1, m2);
  // sprintf(buffer, temp, );

  client.write(buffer);

  return 1;
  }*/

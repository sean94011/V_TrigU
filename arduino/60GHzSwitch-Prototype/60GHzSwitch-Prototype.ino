#include "Arduino.h"
#include "util/delay.h"

void delayMs(int ms) {
  // delay is not in ms after PWM change
  unsigned int delayed = ms << 6;
  delay(delayed);
}

void setup() {
  TCCR0B = (TCCR0B & 0b11111000) | 0x01;
  
  Serial.begin(9600);

  pinMode(5, OUTPUT);
  pinMode(13, OUTPUT); // LED Indicator
  pinMode(2, INPUT);

  Serial.println("Running, perform initial reset:");
  safeMode();  
}

const int delayVal = 1500;
const int ISOdelayVal = 2000;

char rx_byte = 0;
// RBW = 80kHz -> one step = 1/80k.
//const int steps = 10;
//const long rate = (long) (steps / 80.0e3 * 1.0e6); // in microseconds
const long rate = 3000000;
const int duty = 50; // duty cycle for in-chirp modulation

int starttime;
int endtime;

void loop() {
  if (Serial.available() > 0) {    // is a character available?
    rx_byte = Serial.read();
    if (rx_byte < '0' || rx_byte > '9') return;
    
    Serial.print("Number received: ");
    Serial.println(rx_byte);

    Serial.print("Switching rate: ");
    Serial.println(rate);
    
    if (rx_byte=='0') { // turn off the switch
      Serial.println("0 Recv, Ctrl on");
      digitalWrite(5, HIGH);
      digitalWrite(13, HIGH);
    } else if (rx_byte=='1') { //switch to RF1
      Serial.println("1 Recv, Ctrl off");
      safeMode();
    } else if (rx_byte=='3' || digitalRead(4) == HIGH) { // switch constantly
      Serial.println("3 Recv, Constant switching");
      starttime = millis();
      endtime = starttime;

      // Switch tag on/off.
      // 1) Use millis(), not delayMicros(), to keep account of
      // the time needed of switching pins. millis() in Arduino
      // Uno is always a multiple of 4us. Overflow of millis()
      // does not matter since the subtraction always provide
      // the correct result.
      // 2) Use direct pin access instead of digitalWrite/Read,
      // since the former takes ~5us and the latter <1us.
      // 3) Tried figuring out a better solution of turning off
      // the tag, but proto-threading (pseudo-multithreading) does
      // not work with Serial input... probably add a button while
      // designing the new 3D tag?
      unsigned long curr;
      while (digitalRead(2) == LOW) // digitalRead(2) == LOW
      {
        curr = micros();
        PORTD |= _BV(5);
        while (micros() - curr < (rate << 6));
        curr = micros();
        PORTD &= ~_BV(5);
        while (micros() - curr < (rate << 6));
      }
      safeMode();
    }
  }
}

void safeMode() {
  Serial.println("Ctrl Off, safe mode");
  digitalWrite(5, LOW);
  digitalWrite(13, LOW);
  Serial.println("====================");
}

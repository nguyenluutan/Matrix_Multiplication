#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
   FILE *datei;
   int posBLZ = 0;
   int posKontoNr = 0;
   int laengePuffer = 0;
   int n = 0;
   int i = 0;
   char PP[3] = "00";
   char Bankleitzahl[9] = {'\0'};
   char puffer[11] = {'\0'};
   char Filiale[50] = {'\0'};
   char Kontonummer[11] = {'\0'};


   datei = fopen("BLZ.txt","r");   // Datei wird geöffnet (BLZ.txt)

   if(NULL == datei) {
       printf("Datei BLZ:txt konnte nicht geöffnet werden!\n");
       return EXIT_FAILURE;       // Programm wird beendet, wenn Datei nicht geöffent werden konnte
   }

   /* Inhalt der Textdatei zeilenweise ausgeben */

   printf("Liste der Bankfilialen:\n");
   while(!feof(datei)) {
       fgets(Filiale,50,datei);
       printf("%s",Filiale);
   }
   printf("\n");

   do {
       printf("8-stellige Bankleitzahl eingeben:\n");
       scanf("%s",Bankleitzahl);
       posBLZ = strspn(Bankleitzahl,"0123456789");       // Gibt ggf. die Position aus, die keine Ziffer enthält

       printf("10-stellige Kontonummer eingeben:\n");
       scanf("%s,",puffer);
       posKontoNr = strspn(puffer,"0123456789");        // Gibt ggf. die Position aus, die keine Ziffer enthält
       laengePuffer = strlen(puffer);
       if(strlen(Bankleitzahl) < 8) {
           printf("Fehlerhafte Eingabe: Bankleitzahl muss exakt 8 Ziffer haben!\n");
       }
       if((posBLZ >= 0) && (posBLZ < 8)) {
           printf("Fehlerhafte Eingabe bei der BLZ: Keine Ziffer an der Stelle %d eingegeben!\n",++posBLZ);
       }
       if((posKontoNr >= 0) && (posKontoNr < laengePuffer)) {
           printf("%c",Kontonummer[posKontoNr]);
           printf("Fehlerhafte Eingabe bei der Kontonr.: Keine Ziffer an der Stelle %d eingegeben!\n",++posKontoNr);
       }
   }while((strlen(Bankleitzahl) < 8) || ((posBLZ >= 0) && (posBLZ < 8)) || ((posKontoNr >= 0) && (posKontoNr < laengePuffer)));

   rewind(datei);

   n = 10 - laengePuffer;                       // Anzahl der Stellen, die noch benötigt werden
   strncpy(Kontonummer,"0000000000",n);   // Kontonummer wird ggf. vorne mit 0 aufgefüllt
   strncat(Kontonummer,puffer,laengePuffer);


   while(fgets(Filiale,50,datei) != NULL) {
       printf("BLZ: %s\n",Bankleitzahl);
       if(strncmp(Filiale,Bankleitzahl,8)) {
           printf("Filialenname: %s\nZeile %d\n",Filiale,i);
           break;
       }
       i++;
    }
}

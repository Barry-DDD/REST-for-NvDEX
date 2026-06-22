//////////////////////////////////////////////////////////////////////////
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestHitsCombinationProcess.h
///
///              Nov 2016 : Javier Galan
///
//////////////////////////////////////////////////////////////////////////


#ifndef RestCore_TRestHitsCombinationProcess
#define RestCore_TRestHitsCombinationProcess

#include "TString.h"

#include <TRestHitsEvent.h>
#include <TRestSignalEvent.h>
#include <TRestRawSignalEvent.h>
#include "TRestEventProcess.h"

class TRestHitsCombinationProcess:public TRestEventProcess {

    private:

#ifndef __CINT__
        TRestHitsEvent *fInputHitsEvent;//!
	TRestSignalEvent *fInputSignalEvent;//!
	TRestRawSignalEvent *fInputRawSignalEvent;//!
        TRestHitsEvent *fOutputHitsEvent;//!
#endif

        void InitFromConfigFile();

        void Initialize();

    protected:

        Double_t fDistance;
	Double_t fDetZ;


    public:

	bool IsWithinDistance( double x1, double y1, double x2, double y2 );

        void InitProcess();
        void BeginOfEventProcess();
        TRestEvent *ProcessEvent( TRestEvent *eventInput );
        void EndOfEventProcess();
        void EndProcess();
        void LoadDefaultConfig( );

        void LoadConfig( std::string cfgFilename, std::string name = "" );

        void PrintMetadata() {

            BeginPrintProcess();

            std::cout << " det distance : " << fDistance << std::endl;

            EndPrintProcess();
        }

        TString GetProcessName() { return (TString) "hitsCombination"; }

        //Constructor
        TRestHitsCombinationProcess();
        TRestHitsCombinationProcess( char *cfgFileName );
        //Destructor
        ~TRestHitsCombinationProcess();

        ClassDef(TRestHitsCombinationProcess, 1);      // Template for a REST "event process" class inherited from TRestEventProcess
};
#endif

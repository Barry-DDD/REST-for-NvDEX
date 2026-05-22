///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestHitsCombinationProcess.cxx
///
///             Jan 2016:   First concept (Javier Galan)
//
///_______________________________________________________________________________
#include <algorithm>
#include "TRestHitsCombinationProcess.h"
using namespace std;

ClassImp(TRestHitsCombinationProcess)
//______________________________________________________________________________
TRestHitsCombinationProcess::TRestHitsCombinationProcess( )
{
    Initialize();
}

//______________________________________________________________________________
TRestHitsCombinationProcess::TRestHitsCombinationProcess( char *cfgFileName )
{
    Initialize();

    if( LoadConfigFromFile( cfgFileName ) == -1 ) LoadDefaultConfig( );
}

//______________________________________________________________________________
TRestHitsCombinationProcess::~TRestHitsCombinationProcess( )
{
    delete fInputHitsEvent;
    delete fOutputHitsEvent;
    delete fInputSignalEvent;
    delete fInputRawSignalEvent;
}

void TRestHitsCombinationProcess::LoadDefaultConfig( )
{
    SetName( "HitsCombinationProcess" );
    SetTitle( "Default config" );

    fDetZ = fDistance = 0.2;

}

//______________________________________________________________________________
void TRestHitsCombinationProcess::Initialize( )
{
    SetSectionName( this->ClassName() );

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();
    fInputSignalEvent = new TRestSignalEvent();
    fInputRawSignalEvent = new TRestRawSignalEvent();
    fOutputEvent = fOutputHitsEvent;
    //fInputEvent  = fInputSignalEvent;//signal
    fInputEvent  = fInputHitsEvent;
}

void TRestHitsCombinationProcess::LoadConfig( std::string cfgFilename, std::string name )
{

    if( LoadConfigFromFile( cfgFilename, name ) == -1 ) LoadDefaultConfig( );
}

//______________________________________________________________________________
void TRestHitsCombinationProcess::InitProcess()
{
    cout << __PRETTY_FUNCTION__ << endl;
}

//______________________________________________________________________________
void TRestHitsCombinationProcess::BeginOfEventProcess()
{
    fOutputHitsEvent->Initialize();
}
//______________________________________________________________________________
bool TRestHitsCombinationProcess::IsWithinDistance( double x1, double y1, double x2, double y2 ){
	double dis=TMath::Sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
	//if(1){
	if( x1==x2  && TMath::Abs(y1-y2)<3){
		//cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<dis<<endl;
		return true;
	}
	else
		return false;
}
//______________________________________________________________________________
TRestEvent* TRestHitsCombinationProcess::ProcessEvent( TRestEvent *evInput ){
//warning : unsort
//warning : if more than one z,will error
    int nx;
    vector<double> xx;
    vector<double> xz;
    vector<double> xe;
    int ny;
    vector<double> yy;
    vector<double> yz;
    vector<double> ye;

    fInputHitsEvent = (TRestHitsEvent *) evInput;
    //fInputSignalEvent = (TRestSignalEvent *) evInput;//signal
//-----------------find NAN save into 2 vectors---------------
    Int_t number_hits=fInputHitsEvent->GetNumberOfHits();
    //Int_t number_hits=fInputSignalEvent->GetNumberOfSignals();//signal
    cout<<fInputHitsEvent->GetID()<<endl;
    int xq=0;
    //cout<<fInputSignalEvent->GetIntegral()<<endl;//signal
    for( Int_t i = 0; i < number_hits; i++ )
    {/*
        if(TMath::IsNaN(fInputHitsEvent->GetY(i))){
          xx.push_back(fInputHitsEvent->GetX(i));
          xz.push_back(fInputHitsEvent->GetZ(i));
          xe.push_back(fInputHitsEvent->GetEnergy(i));
	  cout<<"xz energy:"<<fInputHitsEvent->GetEnergy(i)<<endl;
	  cout<<fInputHitsEvent->GetX(i)<<" "<<fInputHitsEvent->GetZ(i)<<" "<<fInputHitsEvent->GetEnergy(i)<<endl;
	  continue;
        }
        if(TMath::IsNaN(fInputHitsEvent->GetX(i))){
          yy.push_back(fInputHitsEvent->GetY(i));
          yz.push_back(fInputHitsEvent->GetZ(i));
          ye.push_back(fInputHitsEvent->GetEnergy(i));
          continue;
        }
        fOutputHitsEvent->AddHit( fInputHitsEvent->GetX(i),
                                  fInputHitsEvent->GetY(i),
                                  fInputHitsEvent->GetZ(i),
                                  fInputHitsEvent->GetEnergy(i) );*/

	if(fInputHitsEvent->GetEnergy(i)>2)
	//cout<<"Warning: no_NAN--"<<i<<endl;
	{
		//cout<<"xyz energy:"<<fInputHitsEvent->GetEnergy(i)<<endl;
		xq++;
	}
	
	//cout<<"xyz energy:"<<fInputSignalEvent->GetSignal(i)->GetIntegral()<<endl;//signal
    }
	
	cout<<xq<<endl;

//-------------------------------------------------
//-----------------merge xz---------------
	//cout<<"xz before size:"<<xz.size()<<endl;
	/*for( int i = 0; i < xz.size(); i++ ){
		int merge_xn=0;
		double temp_xz=xz[i];
		for( int j = i+1; j < xz.size(); j++ ){
			if(IsWithinDistance(xx[i],xz[i],xx[j],xz[j])){
				temp_xz+=xz[j];xe[i]+=xe[j];
				xx.erase(xx.begin() + j);
				xz.erase(xz.begin() + j);
				xe.erase(xe.begin() + j);
				merge_xn++;
				j--;
			}
		}
		if(merge_xn!=0) 
			xz[i]=temp_xz/merge_xn;*/
		//need to modify, next point should be an average
		/*if(xz[i+1]>xz[i])		
			xz[i+1] = xz[i] + fDetZ;
		else if(xz[i+1]<xz[i])
			xz[i+1] = xz[i] - fDetZ; */
		//cout<<merge_xn<<"	"<<xx[i]<<"	"<<xz[i]<<"	"<<xe[i]<<endl;   
	//}
	//cout<<"xz after size:"<<xz.size()<<endl;
//-------------------------------------------------
//-----------------merge yz---------------
/*	cout<<yz.size()<<endl;
        for( int i = 0; i < yz.size(); i++ ){
		int merge_yn=0;
		double temp_yz=yz[i];
		for( int j = i+1; j < yz.size(); j++ ){
			//cout<<yy[i]<<"	"<<yz[i]<<"	"<<yy[j]<<"	"<<yz[j]<<endl;
			if(IsWithinDistance(yy[i],yz[i],yy[j],yz[j])){
				temp_yz+=yz[j];ye[i]+=ye[j];
				yy.erase(yy.begin() + j);
				yz.erase(yz.begin() + j);
				ye.erase(ye.begin() + j);
				j--;
			}
		}
		if(merge_yn!=0) 
			yz[i]/=temp_yz/merge_yn;
		if(yz[i+1]<yz[i])		
			yz[i+1] = yz[i] + fDetZ;
		else
			yz[i+1] = yz[i] - fDetZ;    
	}
	cout<<yz.size()<<endl;
//-------------------------------------------------
//----------------- 4 point to merge xz & yz---------------
    if(xz.size()>yz.size()){
	vector <double>::iterator iter = yy.begin();
        for( int h = 0; h < xz.size(); h++ )
        {
	    if(xz[h]>yz[h]-fDistance && xz[h]<yz[h]+fDistance)
	        fOutputHitsEvent->AddHit( xx[h], yy[h], xz[h], xe[h]+ye[h] );
	    else if(xz[h+1]>yz[h]-fDistance && xz[h+1]<yz[h]+fDistance){
		yz.insert( iter+(h), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h], yy[h-1]+(yy[h-1]+yy[h+1])/2, xz[h], xe[h] );
	    }
	    else if(xz[h+2]>yz[h]-fDistance && xz[h+2]<yz[h]+fDistance){
		yz.insert( iter+(h), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h], yy[h-1]+(yy[h-1]+yy[h+2])/3, xz[h], xe[h] );
		yz.insert( iter+(h+1), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h+1], yy[h-1]+(yy[h-1]+yy[h+2])*2/3, xz[h+1], xe[h+1] );
	    }
	    else if(xz[h+3]>yz[h]-fDistance && xz[h+3]<yz[h]+fDistance){
		yz.insert( iter+(h), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h], yy[h-1]+(yy[h-1]+yy[h+3])/4, xz[h], xe[h] );
		yz.insert( iter+(h+1), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h+1], yy[h-1]+(yy[h-1]+yy[h+3])*2/4, xz[h+1], xe[h+1] );
		yz.insert( iter+(h+2), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h+2], yy[h-1]+(yy[h-1]+yy[h+3])*3/4, xz[h+2], xe[h+2] );
	    }
	    else{
		xx.erase(xx.begin() + h);		
		xz.erase(xz.begin() + h);
		xe.erase(xe.begin() + h);
		//cout<<"TRestHitsCombinationProcess: bad xz hits,will delete it!"<<endl;
	    }
	}
    }
    else if(xz.size()<yz.size()){
	vector <double>::iterator iter = xx.begin();
        for( int h = 0; h < yz.size(); h++ )
        {
		   //cout<<h<<"	"<<xz[h]<<"	"<<yz[h]<<endl;

	    if(yz[h]>xz[h]-fDistance && yz[h]<xz[h]+fDistance)
	        fOutputHitsEvent->AddHit( xx[h], yy[h], yz[h], xe[h]+ye[h] );
	    else if(yz[h+1]>xz[h]-fDistance && yz[h+1]<xz[h]+fDistance){
		xz.insert( iter+(h), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h-1]+(xx[h-1]+xx[h+1])/2, yy[h], yz[h], ye[h] );
	    }
	    else if(yz[h+2]>xz[h]-fDistance && yz[h+2]<xz[h]+fDistance){
		xz.insert( iter+(h), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h-1]+(xx[h-1]+xx[h+2])/3, yy[h], yz[h], ye[h] );
		xz.insert( iter+(h+1), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h-1]+(xx[h-1]+xx[h+2])*2/3, yy[h+1], yz[h+1], ye[h+1] );
	    }
	    else if(yz[h+3]>xz[h]-fDistance && yz[h+3]<xz[h]+fDistance){
		xz.insert( iter+(h), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h-1]+(xx[h-1]+xx[h+3])/4, yy[h], xz[h], xe[h] );
		xz.insert( iter+(h+1), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h-1]+(xx[h-1]+xx[h+3])*2/4, yy[h+1], yz[h+1], ye[h+1] );
		xz.insert( iter+(h+2), 1, 0 ); 
		fOutputHitsEvent->AddHit( xx[h-1]+(xx[h-1]+xx[h+3])*3/4, yy[h+2], yz[h+2], ye[h+2] );
	    }
	    else{
		yy.erase(yy.begin() + h);		
		yz.erase(yz.begin() + h);
		ye.erase(ye.begin() + h);
		//cout<<"TRestHitsCombinationProcess: bad yz hits,will delete it!"<<endl;
	    }
	}
    }
            //vector <int>::iterator iElement = find(yz.begin(),yz.begin()+5,xz[h]);
            //if( iElement != yz.end() ){
            //    int dist=distance(yz.begin(),iElement);
            //    fOutputHitsEvent->AddHit( xx[h], yy[h], xz[h], xe[h]+ye[h] );
            //}
            //nx.push_back(h);
    else{
        fOutputHitsEvent= (TRestHitsEvent *) evInput;
	cout<<"cant`t combinate directly"<<endl;
    }*/
//-------------------------------------------------
	fOutputHitsEvent= (TRestHitsEvent *) evInput;
    Int_t initialHits = fInputHitsEvent->GetNumberOfHits();
    Int_t finalHits = fOutputHitsEvent->GetNumberOfHits();

    Int_t initialEnergy = fInputHitsEvent->GetTotalEnergy();
    Int_t finalEnergy = fOutputHitsEvent->GetTotalEnergy();

    //if( this->GetVerboseLevel() == REST_Debug )
    //{
	//cout<<"TRestHitsCombinationProcess :xq : "<<xz.size()<<"	"<<yz.size()<<endl;
        cout << "TRestHitsCombinationProcess : Initial number of hits : " << initialHits << endl;
        cout << "TRestHitsCombinationProcess : Final number of hits : " << finalHits << endl;
        cout << "TRestHitsCombinationProcess : Initial energy of hits : " << initialEnergy << endl;
        cout << "TRestHitsCombinationProcess : Final energy of hits : " << finalEnergy << endl;
    //}

    return fOutputHitsEvent;
}

//______________________________________________________________________________
void TRestHitsCombinationProcess::EndOfEventProcess()
{

}

//______________________________________________________________________________
void TRestHitsCombinationProcess::EndProcess()
{
}

//______________________________________________________________________________
void TRestHitsCombinationProcess::InitFromConfigFile( )
{
    fDetZ = GetDblParameterWithUnits(  "DetZ" );
    fDistance = GetDblParameterWithUnits(  "Distance" );
}

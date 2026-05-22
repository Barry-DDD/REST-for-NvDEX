void GetG4XYZ(){

	TFile* f=new TFile("example_output.root");
	//TFile* f=new TFile("ovbb.root");
	TTree* t=(TTree*)f->Get("EventTree");
	TRestG4Event* g4=new TRestG4Event();
	t->SetBranchAddress("TRestG4EventBranch",&g4);


	//TRestHits hit=*(g4->GetTrack(0)->GetHits());
	TRestHits hit=g4->GetHits();
	//ofstream w("ovbb.txt");
	//ofstream w("../Bayes/electron_400MeV_2.txt");

	for(int i=0;i<t->GetEntries();i++){
		t->GetEntry(i);
		cout<<g4->GetPrimaryEventEnergy(0)/1000<<" "<<g4->GetNumberOfHits()<<endl;
	}

	exit(0);
}

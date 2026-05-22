//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4Geantino.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "Randomize.hh"
#include <cmath>
#include <algorithm>

#include <TRestG4Event.h>
#include <TRestG4Metadata.h>

extern TRestG4Metadata *restG4Metadata;
extern TRestG4Event *restG4Event;

extern Int_t biasing;

Int_t face = 0;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::PrimaryGeneratorAction(DetectorConstruction *pDetector)
    : G4VUserPrimaryGeneratorAction(), fParticleGun(0), fDetector(pDetector)
{
  G4int n_particle = 1;
  fParticleGun = new G4ParticleGun(n_particle);

  nCollections = restG4Metadata->GetPrimaryGenerator().GetNumberOfCollections();

  nBiasingVolumes = restG4Metadata->GetNumberOfBiasingVolumes();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fParticleGun; }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent)
{
  if (restG4Metadata->GetVerboseLevel() >= REST_Info)
    cout << "Primary generation..." << endl;
  // We have to initialize here and not in start of the event because
  // GeneratePrimaries is called first, and we want to store event origin and
  // position inside
  // we should have already written the information from previous event to disk
  // (in endOfEventAction)
  restG4Event->Initialize();

  // If there are particle collections stored is because we are using a
  // generator from file
  if (nCollections > 0)
  {
    Int_t rndCollection = (Int_t)(G4UniformRand() * nCollections);

    restG4Metadata->SetParticleCollection(rndCollection);
  }

  Int_t nParticles = restG4Metadata->GetNumberOfPrimaries();

  // Position is common for all particles

  for (int j = 0; j < nParticles; j++)
  {
    // ParticleDefinition should be always declared first
    SetParticleDefinition(j);
    SetParticlePosition(j);

    // Particle Direction must be always set before energy
    SetParticleEnergy(j);

    SetParticleDirection(j);

    SetParticleTime(j);

    fParticleGun->GeneratePrimaryVertex(anEvent);
  }
}
void PrimaryGeneratorAction::SetParticleTime(int n)
{
  double time = 0;
  std::string type = (std::string)restG4Metadata->GetGeneratorType();

  if (type == "readfromfile")
  {
    time = restG4Metadata->GetParticleSource(n).GetTime(); // 单位：s
  }

  time *= 1e9; // 转换为ns， Geant4 默认单位是 ns
  fParticleGun->SetParticleTime(time);
  // std::cout << "Set particle time for particle " << n << " : " << time << " ns" << std::endl;
  if (restG4Metadata->GetVerboseLevel() >= REST_Debug)
  {
    std::cout << "Set particle time for particle " << n << " : " << time << " ns" << std::endl;
  }
}

//_____________________________________________________________________________
G4ParticleDefinition *PrimaryGeneratorAction::SetParticleDefinition(int n)
{
  string particleName =
      (string)restG4Metadata->GetParticleSource(n).GetParticleName();

  Double_t eenergy =
      (double)restG4Metadata->GetParticleSource(n).GetExcitationLevel();

  Int_t charge = restG4Metadata->GetParticleSource(n).GetParticleCharge();

  if (restG4Metadata->GetVerboseLevel() >= REST_Debug)
  {
    cout << "-- Debug : Searching for : " << endl;
    cout << "-- Debug : particle name : " << particleName << endl;
    cout << "-- Debug : particle charge : " << charge << endl;
    cout << "-- Debug : particle excited energy : " << eenergy << endl;
  }

  G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition *particle = particleTable->FindParticle(particleName);

  if ((particle == NULL))
  {
    // There might be a better way to do this
    for (int Z = 1; Z <= 110; Z++)
      for (int A = 2 * Z; A <= 3 * Z; A++)
      {
        //   cout << "Ion name : " << G4IonTable::GetIonTable()->GetIonName ( Z,
        //   A ) << endl;
        if (particleName == G4IonTable::GetIonTable()->GetIonName(Z, A))
        {
          particle = G4IonTable::GetIonTable()->GetIon(Z, A, eenergy);
          fParticleGun->SetParticleCharge(charge);
          cout << "Found ion: " << particleName << " Z " << Z << " A " << A
               << " excited energy " << eenergy << endl;
        }
      }
  }

  fParticleGun->SetParticleDefinition(particle);

  restG4Event->SetPrimaryEventParticleName(particleName);

  return particle;
}

void PrimaryGeneratorAction::SetParticleDirection(int n)
{
  G4ThreeVector direction;

  string type =
      (string)restG4Metadata->GetParticleSource(n).GetAngularDistType();

  // TODO make this kind of string keyword comparisons case insensitive?
  if (type == "isotropic")
  {
    if ((string)restG4Metadata->GetGeneratorType() == "virtualBox")
    {
      if (face == 0)
        direction.set(0, -1, 0);
      if (face == 1)
        direction.set(0, 1, 0);
      if (face == 2)
        direction.set(-1, 0, 0);
      if (face == 3)
        direction.set(1, 0, 0);
      if (face == 4)
        direction.set(0, 0, -1);
      if (face == 5)
        direction.set(0, 0, 1);

      Double_t theta = GetCosineLowRandomThetaAngle();
      // recording the primaries distribution
      G4ThreeVector referenceOrigin = direction;

      // We rotate the origin direction by the angular distribution angle
      G4ThreeVector orthoVector = direction.orthogonal();
      direction.rotate(theta, orthoVector);

      // We rotate a random angle along the original direction
      Double_t randomAngle = G4UniformRand() * 2 * M_PI;
      direction.rotate(randomAngle, referenceOrigin);
    }
    else if ((string)restG4Metadata->GetGeneratorType() == "virtualSphere")
    {
      direction = -fParticleGun->GetParticlePosition().unit();

      Double_t theta = GetCosineLowRandomThetaAngle();

      G4ThreeVector referenceOrigin = direction;

      // We rotate the origin direction by the angular distribution angle
      G4ThreeVector orthoVector = direction.orthogonal();
      direction.rotate(theta, orthoVector);

      // We rotate a random angle along the original direction
      Double_t randomAngle = G4UniformRand() * 2 * M_PI;
      direction.rotate(randomAngle, referenceOrigin);
    }
    else
    {
      direction = GetIsotropicVector();
    }
  }

  else if (type == "TH1D")
  {
    Double_t angle = 0;
    Double_t value = G4UniformRand() * (fAngularDistribution->Integral());
    Double_t sum = 0;

    Double_t deltaAngle = fAngularDistribution->GetBinCenter(2) -
                          fAngularDistribution->GetBinCenter(1);
    for (int bin = 1; bin <= fAngularDistribution->GetNbinsX(); bin++)
    {
      sum += fAngularDistribution->GetBinContent(bin);

      if (sum >= value)
      {
        angle = fAngularDistribution->GetBinCenter(bin) +
                deltaAngle * (0.5 - G4UniformRand());
        break;
      }
    }

    // A vector pointing to the origin (virtualSphere )
    direction = -fParticleGun->GetParticlePosition().unit();

    if (direction.x() == 0 && direction.y() == 0 && direction.z() == 0)
    {
      cout << "----------------------------------------------------------------"
              "-----"
           << endl;
      cout << "REST WARNNING : Particle being launched from the ORIGIN!! Wrong "
              "momentum direction!"
           << endl;
      cout << "Setting direction to (1,0,0)" << endl;
      cout << "REST angular distribution is just implemented for virtualBox "
              "and virtualSphere"
           << endl;
      cout << "Other spatial distributions can be set but it will launch the "
              "event\n with a distribution direction to the origin of "
              "coordinates"
           << endl;
      cout << "----------------------------------------------------------------"
              "-----"
           << endl;
      direction.set(1, 0, 0);
    }

    if ((string)restG4Metadata->GetGeneratorType() == "virtualBox")
    {
      if (face == 0)
        direction.set(0, -1, 0);
      if (face == 1)
        direction.set(0, 1, 0);
      if (face == 2)
        direction.set(-1, 0, 0);
      if (face == 3)
        direction.set(1, 0, 0);
      if (face == 4)
        direction.set(0, 0, -1);
      if (face == 5)
        direction.set(0, 0, 1);
    }

    G4ThreeVector referenceOrigin = direction;

    // We rotate the origin direction by the angular distribution angle
    G4ThreeVector orthoVector = direction.orthogonal();
    direction.rotate(angle, orthoVector);

    // We rotate a random angle along the radial direction
    Double_t randomAngle = G4UniformRand() * 2 * M_PI;
    direction.rotate(randomAngle, referenceOrigin);

    //       G4cout << "Angle  " << direction.angle( referenceOrigin ) << "
    //       should be = to " << angle << G4endl;
  }
  else if (type == "flux")
  {
    TVector3 v = restG4Metadata->GetParticleSource(n).GetDirection();

    v = v.Unit();

    direction.set(v.X(), v.Y(), v.Z());
  }
  else if (type == "planeSector")
  {
    int i = n; // source index

    // 下面这些 getter 你后续再去 TRestParticleSource 里补定义
    TVector3 normal = restG4Metadata->GetParticleSource(i).GetPlaneNormal().Unit();
    TVector3 ref = restG4Metadata->GetParticleSource(i).GetPlaneRef();

    double phiMin = restG4Metadata->GetParticleSource(i).GetPhiMin(); // rad
    double phiMax = restG4Metadata->GetParticleSource(i).GetPhiMax(); // rad
    if (phiMin > phiMax)
      std::swap(phiMin, phiMax);

    // ref 投影到平面内，保证 dir 在该平面
    ref = ref - (ref.Dot(normal)) * normal;
    if (ref.Mag2() < 1e-12)
      ref = normal.Orthogonal();

    TVector3 uHat = ref.Unit();
    TVector3 vHat = normal.Cross(uHat).Unit();

    // phi 在 [phiMin, phiMax] 均匀
    double u = G4UniformRand(); // 如果你项目不是G4，用你现有的均匀随机
    double phi = phiMin + u * (phiMax - phiMin);

    TVector3 dir = std::cos(phi) * uHat + std::sin(phi) * vHat;
    dir = dir.Unit();
    direction.set(dir.X(), dir.Y(), dir.Z());
  }
  else if (type == "angle")
  {
    auto src = restG4Metadata->GetParticleSource(n);

    double thetaMin = src.GetAngleThetaMin();
    double thetaMax = src.GetAngleThetaMax();
    double phiMin = src.GetAnglePhiMin();
    double phiMax = src.GetAnglePhiMax();

    // ---- 均匀立体角采样：cos(theta) 均匀 ----
    double cosMin = std::cos(thetaMax);
    double cosMax = std::cos(thetaMin);
    if (cosMin > cosMax)
      std::swap(cosMin, cosMax);
    double u = cosMin + (cosMax - cosMin) * G4UniformRand();
    double phi = phiMin + (phiMax - phiMin) * G4UniformRand();
    double theta = std::acos(u);

    double st = std::sin(theta);
    double x = st * std::cos(phi);
    double y = st * std::sin(phi);
    double z = std::cos(theta);

    direction.set(x, y, z);
  }

  else if (type == "backtoback")
  {
    // This should never crash. In TRestG4Metadata we have defined that if the
    // first source is backtoback we set it to isotropic
    TVector3 v = restG4Event->GetPrimaryEventDirection(n - 1);
    v = v.Unit();

    direction.set(-v.X(), -v.Y(), -v.Z());
  }
  else
  {
    G4cout << "WARNING! Generator angular distribution was not recognized. "
              "Launching particle to (1,0,0)"
           << G4endl;
  }

  // storing the direction in TRestG4Event class
  TVector3 eventDirection(direction.x(), direction.y(), direction.z());
  restG4Event->SetPrimaryEventDirection(eventDirection);

  if (restG4Metadata->GetVerboseLevel() >= REST_Debug)
  {
    cout << "Event direction has been set : " << endl;
    cout << "(" << restG4Event->GetPrimaryEventDirection(0).X() << ", "
         << restG4Event->GetPrimaryEventDirection(0).Y() << ", "
         << restG4Event->GetPrimaryEventDirection(0).Z() << ")" << endl;
  }

  // setting particle direction
  fParticleGun->SetParticleMomentumDirection(direction);
}

//_____________________________________________________________________________
void PrimaryGeneratorAction::SetParticleEnergy(int n)
{
  string type =
      (string)restG4Metadata->GetParticleSource(n).GetEnergyDistType();

  Double_t energy = 0;

  if (type == "mono")
  {
    energy = restG4Metadata->GetParticleSource(n).GetEnergy() * keV;
  }
  else if (type == "flat")
  {
    TVector2 enRange = restG4Metadata->GetParticleSource(n).GetEnergyRange();

    energy =
        ((enRange.Y() - enRange.X()) * G4UniformRand() + enRange.X()) * keV;
  }
  else if (type == "TH1D")
  {
    Double_t value = G4UniformRand() * fSpectrumIntegral;
    Double_t sum = 0;
    Double_t deltaEnergy =
        fSpectrum->GetBinCenter(2) - fSpectrum->GetBinCenter(1);
    for (int bin = startEnergyBin; bin <= endEnergyBin; bin++)
    {
      sum += fSpectrum->GetBinContent(bin);

      if (sum > value)
      {
        energy = energyFactor *
                 (Double_t)(fSpectrum->GetBinCenter(bin) +
                            deltaEnergy * (0.5 - G4UniformRand())) *
                 keV;
        break;
      }
    }
  }
  else
  {
    G4cout << "WARNING! Energy distribution type was not recognized. Setting "
              "energy to 1keV"
           << G4endl;
    energy = 1 * keV;
  }
  // cout << "Setting particle Energy : " << energy / keV << " keV" << endl;
  fParticleGun->SetParticleEnergy(energy);

  restG4Event->SetPrimaryEventEnergy(energy / keV);
  // if (restG4Metadata->GetVerboseLevel() >= REST_Debug)
  // cout << "Setting particle Energy : " << energy / keV << " keV" << endl;
}

//_____________________________________________________________________________
void PrimaryGeneratorAction::SetParticlePosition(int n)
{
  double x = 0, y = 0, z = 0;
  string type = (string)restG4Metadata->GetGeneratorType();

  // TODO make this kind of string keyword comparisons case insensitive?
  if (type == "volume")
  {
    double xMin = fDetector->GetBoundingX_min();
    double xMax = fDetector->GetBoundingX_max();
    double yMin = fDetector->GetBoundingY_min();
    double yMax = fDetector->GetBoundingY_max();
    double zMin = fDetector->GetBoundingZ_min();
    double zMax = fDetector->GetBoundingZ_max();

    do
    {
      x = xMin + (xMax - xMin) * G4UniformRand();
      y = yMin + (yMax - yMin) * G4UniformRand();
      z = zMin + (zMax - zMin) * G4UniformRand();
    } while (fDetector->GetGeneratorSolid()->Inside(G4ThreeVector(x, y, z)) !=
             kInside);

    x = x + fDetector->GetGeneratorTranslation().x();
    y = y + fDetector->GetGeneratorTranslation().y();
    z = z + fDetector->GetGeneratorTranslation().z();
  }
  else if (type == "surface")
  {
    // TODO there is a problem, probably with G4 function GetPointOnSurface
    // It produces a point on the surface but it is not uniformly distributed
    // May be it is just an OPENGL drawing issue?

    G4ThreeVector position =
        fDetector->GetGeneratorSolid()->GetPointOnSurface();

    x = position.x();
    y = position.y();
    z = position.z();

    x = x + fDetector->GetGeneratorTranslation().x();
    y = y + fDetector->GetGeneratorTranslation().y();
    z = z + fDetector->GetGeneratorTranslation().z();
  }
  else if (type == "point")
  {
    TVector3 position = restG4Metadata->GetGeneratorPosition();

    x = position.X();
    y = position.Y();
    z = position.Z();
  }
  else if (type == "virtualWall")
  {
    Double_t side = restG4Metadata->GetGeneratorSize();

    x = side * (G4UniformRand() - 0.5);
    y = side * (G4UniformRand() - 0.5);

    G4ThreeVector rndPos = G4ThreeVector(x, y, 0);
    rndPos.rotateX(M_PI * restG4Metadata->GetGeneratorRotation().X() / 180.);
    rndPos.rotateY(M_PI * restG4Metadata->GetGeneratorRotation().Y() / 180.);
    rndPos.rotateZ(M_PI * restG4Metadata->GetGeneratorRotation().Z() / 180.);

    TVector3 center = restG4Metadata->GetGeneratorPosition();

    x = rndPos.x() + center.X();
    y = rndPos.y() + center.Y();
    z = rndPos.z() + center.Z();
  }
  else if (type == "virtualCircleWall")
  {
    Double_t radius = restG4Metadata->GetGeneratorSize();

    do
    {
      x = 2 * radius * (G4UniformRand() - 0.5);
      y = 2 * radius * (G4UniformRand() - 0.5);
      //       cout << "x : " << x << " y : " << y << endl;
    } while (x * x + y * y > radius * radius);

    G4ThreeVector rndPos = G4ThreeVector(x, y, 0);
    rndPos.rotateX(M_PI * restG4Metadata->GetGeneratorRotation().X() / 180.);
    rndPos.rotateY(M_PI * restG4Metadata->GetGeneratorRotation().Y() / 180.);
    rndPos.rotateZ(M_PI * restG4Metadata->GetGeneratorRotation().Z() / 180.);

    TVector3 center = restG4Metadata->GetGeneratorPosition();

    x = rndPos.x() + center.X();
    y = rndPos.y() + center.Y();
    z = rndPos.z() + center.Z();
  }
  else if (type == "virtualSphere")
  {
    G4ThreeVector rndPos = GetIsotropicVector();

    Double_t radius = restG4Metadata->GetGeneratorSize();

    TVector3 center = restG4Metadata->GetGeneratorPosition();

    x = radius * rndPos.x() + center.X();
    y = radius * rndPos.y() + center.Y();
    z = radius * rndPos.z() + center.Z();
  }
  else if (type == "virtualCylinder")
  {
    Double_t angle = 2 * M_PI * G4UniformRand();

    Double_t radius = restG4Metadata->GetGeneratorSize();
    Double_t length = restG4Metadata->GetGeneratorLength();

    x = radius * cos(angle);
    y = radius * sin(angle);
    z = length * (G4UniformRand() - 0.5);

    G4ThreeVector rndPos = G4ThreeVector(x, y, z);
    rndPos.rotateX(M_PI * restG4Metadata->GetGeneratorRotation().X() / 180.);
    rndPos.rotateY(M_PI * restG4Metadata->GetGeneratorRotation().Y() / 180.);
    rndPos.rotateZ(M_PI * restG4Metadata->GetGeneratorRotation().Z() / 180.);

    TVector3 center = restG4Metadata->GetGeneratorPosition();

    x = rndPos.x() + center.X();
    y = rndPos.y() + center.Y();
    z = rndPos.z() + center.Z();
  }
  else if (type == "virtualBox")
  {
    Double_t side = restG4Metadata->GetGeneratorSize();

    x = side * (G4UniformRand() - 0.5);
    y = side * (G4UniformRand() - 0.5);

    G4ThreeVector rndPos = G4ThreeVector(x, y, 0);

    Double_t rndOrientation = 3 * G4UniformRand();
    Double_t rndFace = G4UniformRand();
    if (rndOrientation <= 1)
    {
      // Event is in plane XZ
      rndPos.rotateX(M_PI / 2.);
      x = rndPos.x();
      y = rndPos.y();
      z = rndPos.z();

      // Some rounding error after rotate
      y = 0;

      // cout << "Event in plane XZ. x : " << x << " y : " << y << " z : " << z
      // << endl;

      if (rndFace <= 0.5)
      {
        y = y + side / 2.;
        face = 0;
      }
      else
      {
        y = y - side / 2.;
        face = 1;
      }
    }
    else if (rndOrientation <= 2)
    {
      // Event is in plane YZ
      rndPos.rotateY(M_PI / 2.);
      x = rndPos.x();
      y = rndPos.y();
      z = rndPos.z();

      // Some rounding error after rotate
      x = 0;

      //           cout << "Event in plane YZ. x : " << x << " y : " << y << " z
      //           : " << z << endl;

      if (rndFace <= 0.5)
      {
        x = x + side / 2.;
        face = 2;
      }
      else
      {
        x = x - side / 2.;
        face = 3;
      }
    }
    else
    {
      // Event is in plane XY
      x = rndPos.x();
      y = rndPos.y();
      z = rndPos.z();

      //         cout << "Event in plane XY. x : " << x << " y : " << y << " z :
      //         " << z << endl;

      //      cout << "z : " << y;
      if (rndFace <= 0.5)
      {
        z = z + side / 2.;
        face = 4;
      }
      else
      {
        z = z - side / 2.;
        face = 5;
      }
    }

    TVector3 center = restG4Metadata->GetGeneratorPosition();

    // G4cout << "X : " << x << " Y : " << y << " Z : " << z << G4endl;

    x = x + center.X();
    y = y + center.Y();
    z = z + center.Z();
  }
  else if (type == "readfromfile")
  {
    fCurrentParticle = restG4Metadata->GetParticleSource(n);

    x = fCurrentParticle.GetPosX();
    y = fCurrentParticle.GetPosY();
    z = fCurrentParticle.GetPosZ();
  }

  else
  {
    G4cout << "WARNING! Generator type was not recognized. Launching particle "
              "from origin (0,0,0)"
           << G4endl;
  }

  // storing the direction in TRestG4Event class
  TVector3 eventPosition(x, y, z);
  restG4Event->SetPrimaryEventOrigin(eventPosition);
  // cout << "Event origin has been set : " << endl;
  // cout << "(" << restG4Event->GetPrimaryEventOrigin().X() << ", "
  //      << restG4Event->GetPrimaryEventOrigin().Y() << ", "
  //      << restG4Event->GetPrimaryEventOrigin().Z() << ")" << endl;
  if (restG4Metadata->GetVerboseLevel() >= REST_Debug)
  {
    cout << "Event origin has been set : " << endl;
    cout << "(" << restG4Event->GetPrimaryEventOrigin().X() << ", "
         << restG4Event->GetPrimaryEventOrigin().Y() << ", "
         << restG4Event->GetPrimaryEventOrigin().Z() << ")" << endl;
  }

  // setting particle position
  fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
}

//_____________________________________________________________________________
G4ThreeVector PrimaryGeneratorAction::GetIsotropicVector()
{
  G4double a, b, c;
  G4double n;
  do
  {
    a = (G4UniformRand() - 0.5) / 0.5;
    b = (G4UniformRand() - 0.5) / 0.5;
    c = (G4UniformRand() - 0.5) / 0.5;
    n = a * a + b * b + c * c;
  } while (n > 1 || n == 0.0);

  n = std::sqrt(n);
  a /= n;
  b /= n;
  c /= n;
  return G4ThreeVector(a, b, c);
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//
Double_t PrimaryGeneratorAction::GetAngle(G4ThreeVector x, G4ThreeVector y)
{
  Double_t angle = y.angle(x);

  return angle;
}

Double_t PrimaryGeneratorAction::GetCosineLowRandomThetaAngle()
{
  // We obtain an angle with a cos(theta)*sin(theta) distribution
  Double_t value = G4UniformRand();
  double dTheta = 0.01;
  for (double theta = 0; theta < M_PI / 2; theta = theta + dTheta)
  {
    // sin(theta)^2 is the integral of sin(theta)*cos(theta)
    if (sin(theta) * sin(theta) >= value)
    {
      if (theta > 0)
        return theta - dTheta * (0.5 - G4UniformRand());
      else
        return theta;
    }
  }
  return M_PI / 2;
}

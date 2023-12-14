from fastapi import FastAPI
from rdkit import Chem
from rdkit.Chem import AllChem
import uvicorn
from rdkit.Chem import Draw
from fastapi.responses import HTMLResponse
from fastapi import File, UploadFile
from rdkit.Chem import rdMolAlign
from io import BytesIO
import py3Dmol
import base64

app = FastAPI()

@app.get("/")
def home():
    return "Ana sayfaya hoş geldiniz!"

@app.get("/mol/{smiles}", response_model=dict)
def read_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol)

    atoms = []
    bonds = []
    for atom in mol.GetAtoms():
        atoms.append({
            'serial': atom.GetIdx(),
            'name': atom.GetSymbol(),
            'elem': atom.GetSymbol(),
            'mass_magnitude': atom.GetMass(),
            'residue_index': atom.GetMonomerInfo().GetResidueNumber() if atom.GetMonomerInfo() is not None else None,
            'residue_name': atom.GetMonomerInfo().GetResidueName() if atom.GetMonomerInfo() is not None else None,
            'chain': atom.GetMonomerInfo().GetChainId() if atom.GetMonomerInfo() is not None else None,
            'positions': list(mol.GetConformer().GetAtomPosition(atom.GetIdx())),
        })
    for bond in mol.GetBonds():
        bonds.append({
            'atom1_index': bond.GetBeginAtomIdx(),
            'atom2_index': bond.GetEndAtomIdx(),
            'bond_order': bond.GetBondTypeAsDouble(),
        })

    return {'atoms': atoms, 'bonds': bonds}

@app.get("/draw/{smiles}", response_class=HTMLResponse)
def draw_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(400, 400), kekulize=True, wedgeBonds=True, wedgeBondsWidth=0.5)
    
    byte_io = BytesIO()
    img.save(byte_io, format='PNG')
    byte_io.seek(0)
    
    base64_img = base64.b64encode(byte_io.read()).decode('ascii')
    
    html = f'<img src="data:image/png;base64,{base64_img}" alt="Molecule">'
    
    return HTMLResponse(content=html)

@app.get("/draw3d/{smiles}")
async def draw_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMolecule(mol)

    block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=400, height=400)

    viewer.addModel(block, "mol")

    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('white')

    img = viewer.capture_image()

    base64_img = base64.b64encode(img)
    
    return {"base64_img": base64_img.decode()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

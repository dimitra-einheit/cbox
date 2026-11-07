# Setting up dronemap on the Windows PC (i7-8700K, 32 GB, Vega 64)

The PC can **plan flights, process photos with OpenDroneMap, and run the terrain and pond tools**.
It can't train Gaussian splats at a usable speed. OpenSplat needs CUDA (NVIDIA) or recent AMD ROCm,
and the Vega 64 is no longer supported by either. Do splats on the Mac.

The Mac (64 GB) is also faster for big ODM jobs. Use the PC as a second processing machine, or for
storage: its 8 TB drive is a good home for raw photos.

## 1. Install the software (one time)

1. **Python 3.12**: download from https://www.python.org/downloads/windows/ and run the installer.
   Tick **"Add python.exe to PATH"**.
2. **Docker Desktop**: download from https://www.docker.com/products/docker-desktop/ and install it
   with the **WSL 2** option. Restart the PC when asked. Open Docker Desktop once, accept the terms
   (signing in is optional), and wait until it shows *Engine running*.
3. In Docker Desktop, open **Settings > Resources > WSL integration** (or *Advanced*) and leave the
   defaults. WSL 2 can use up to half of the PC's memory (16 GB), which handles a few hundred photos.
   To allow more, create the file `C:\Users\<you>\.wslconfig` with:
   ```
   [wsl2]
   memory=26GB
   processors=12
   swap=8GB
   ```
   then run `wsl --shutdown` in PowerShell and restart Docker Desktop.
4. Copy the `dronemap` folder to the PC (for example `C:\dronemap`).

## 2. Create the Python environment

Open **PowerShell** in the `dronemap` folder (Shift + right-click > *Open PowerShell window here*):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker pull opendronemap/odm:latest
.\dronemap.cmd doctor
```

`doctor` should show OK for Python packages and Docker. A warning about OpenSplat is expected on the PC.

## 3. Use it

Run commands the same way as on the Mac, but with `.\dronemap.cmd`:

```powershell
.\dronemap.cmd plan C:\maps\bagrod.kml --preset survey
.\dronemap.cmd process D:\drone\projects\bagrod --preset survey
.\dronemap.cmd site D:\drone\projects\bagrod
.\dronemap.cmd pond D:\drone\projects\bagrod --outline C:\maps\pond.kml --depth 2.5,3,3.5
```

Keep project folders on a local NTFS drive (the 8 TB disk is fine). Don't use network shares: Docker
reads them very slowly. Folder names must not contain spaces.

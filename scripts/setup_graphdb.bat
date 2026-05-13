@echo off
setlocal EnableDelayedExpansion

:: ==============================================================================
:: GraphDB System-Wide Setup for Windows
:: Installs to %USERPROFILE%\graphdb-server
:: ==============================================================================

set GRAPHDB_VERSION=11.3.3
set INSTALL_DIR=%USERPROFILE%\graphdb-server
set ZIP_FILE=%TEMP%\graphdb-dist.zip
set DOWNLOAD_URL="https://download.ontotext.com/owlim/0521929f-94ab-4ac0-adce-84fc0426b69e/graphdb-11.3.3-dist.zip?_gl=1*15600kb*_ga*MTE2MzA3NTI2OS4xNzc4MTMyOTEw*_ga_HGSKWBWCRK*czE3Nzg2OTk1ODMkbzMkZzEkdDE3Nzg3MDAxMTYkajU5JGwwJGgw"

echo === Starting System-Wide GraphDB Setup ===

:: 1. Download and Extract to the User Profile
if not exist "%INSTALL_DIR%" (
    echo [*] Downloading GraphDB %GRAPHDB_VERSION% to Temp... 
    powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL:'='%' -OutFile '%ZIP_FILE%'"
    
    echo [*] Extracting to User Profile...
    powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%USERPROFILE%' -Force"
    
    :: Rename the folder to the clean path
    rename "%USERPROFILE%\graphdb-%GRAPHDB_VERSION%" "graphdb-server"
    del "%ZIP_FILE%"
) else (
    echo [*] GraphDB is already installed at %INSTALL_DIR%.
)

:: 2. Inject License File
if exist "..\graphdb.license" (
    echo [*] Found graphdb.license. Injecting into system config...
    copy /Y "..\graphdb.license" "%INSTALL_DIR%\conf\graphdb.license" >nul
)

:: 3. Start GraphDB in the Background
echo [*] Launching System GraphDB...
start /B "" "%INSTALL_DIR%\bin\graphdb.bat" > "%INSTALL_DIR%\startup.log" 2>&1

echo === Installation Complete! ===
echo GraphDB is permanently installed at: %INSTALL_DIR%
echo Please wait 15 seconds, then open: http://localhost:7200
pause
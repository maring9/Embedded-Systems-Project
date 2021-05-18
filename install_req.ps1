<#
This script installs the requirements in the virtual environment
#>

if (Test-Path env:VIRTUAL_ENV) {
	if ( [System.IO.File]::Exists("requirements.txt") ) {
		pip install -r requirements.txt
	}
	else {
		Write-Error "The current directory does not contain requierements.txt file"
		Write-Warning "You will need to install all the encessary packages manually"
	}
}
else {
	Write-Error "Currently not in a virtual environment. Requierements will not be installed."
}
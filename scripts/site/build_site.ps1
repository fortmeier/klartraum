# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Assembles the landing page and documentation in _site/, like build_site.sh.
# SPHINX_BUILD overrides the executable; otherwise prefer docs/.venv, then PATH.
param(
    [string[]]$SphinxArgs = @()
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$siteOutput = [System.IO.Path]::GetFullPath((Join-Path $repoRoot '_site'))
$sphinxBuild = $env:SPHINX_BUILD
if ([string]::IsNullOrWhiteSpace($sphinxBuild)) {
    $sphinxBuild = Join-Path $repoRoot 'docs/.venv/Scripts/sphinx-build.exe'
    if (-not (Test-Path -LiteralPath $sphinxBuild -PathType Leaf)) {
        $sphinxBuild = 'sphinx-build'
    }
}

# Check dependencies before replacing the previous output.
$null = Get-Command $sphinxBuild -CommandType Application -ErrorAction Stop
$null = Get-Command doxygen -CommandType Application -ErrorAction Stop

# Only remove the repository's own output directory, never a directory link.
if ((Split-Path -Parent $siteOutput) -ne $repoRoot) {
    throw "Site output is outside the repository: $siteOutput"
}
if (Test-Path -LiteralPath $siteOutput) {
    if ((Get-Item -LiteralPath $siteOutput -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        throw "Site output must not be a directory link: $siteOutput"
    }
    Remove-Item -LiteralPath $siteOutput -Recurse -Force
}
$null = New-Item -ItemType Directory -Path $siteOutput
Get-ChildItem -LiteralPath (Join-Path $repoRoot 'site') -Force |
    Copy-Item -Destination $siteOutput -Recurse -Force

Push-Location -LiteralPath $repoRoot
try {
    & $sphinxBuild -W --keep-going -q @SphinxArgs docs (Join-Path $siteOutput 'docs')
    if ($LASTEXITCODE -ne 0) {
        throw "Sphinx failed with exit code $LASTEXITCODE."
    }
    # Keep GitHub Pages from processing the output with Jekyll.
    $null = New-Item -ItemType File -Path (Join-Path $siteOutput '.nojekyll') -Force
    Write-Output 'site assembled in _site/'
}
finally {
    Pop-Location
}

param(
    [Parameter(Mandatory=$true)] [ValidateSet('Flags','Emotions','Yeast')] [string] $Dataset,
    [Parameter(Mandatory=$true)] [string] $Arff,
    [string] $Xml = '',
    [switch] $Track,
    [int[]] $Seeds = @(42, 43, 44)
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Arff)) { throw "ARFF file not found: $Arff" }
if ($Xml -and -not (Test-Path $Xml)) { throw "XML file not found: $Xml" }
foreach ($seed in $Seeds) {
    foreach ($model in @('stage2', 'stage3', 'majority')) {
        $arguments = @('.\experiments\run_paper_benchmark.py', '--dataset', $Dataset, '--arff', $Arff, '--model', $model, '--seed', "$seed")
        if ($Xml) { $arguments += @('--xml', $Xml) }
        if ($Track) { $arguments += @('--wandb', '--project', 'mlhat17-reproduction') }
        Write-Host "RUN $Dataset / $model / seed $seed" -ForegroundColor Cyan
        & python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Benchmark command failed: $Dataset / $model / seed $seed" }
    }
}
python .\experiments\summarize_lab1.py
if ($LASTEXITCODE -ne 0) { throw 'Benchmark summary failed' }

$choice = Read-Host "Run with (g)unicorn or (u)vicorn?"

$sentinelFile = ".docker_built"
$needsBuild = -not (Test-Path $sentinelFile)

if ($choice -eq "u") {
    $uvChoice = Read-Host "Run uvicorn (l)ocal or (d)ocker?"

    if ($uvChoice -eq "l") {
        # Infra only (no app) — app runs on host so audio devices work
        if ($needsBuild) {
            docker compose -f docker-compose.yaml up --build -d --scale app=0
        } else {
            docker compose -f docker-compose.yaml up -d --scale app=0
        }

        if ($LASTEXITCODE -eq 0) {
            if ($needsBuild) {
                New-Item -ItemType File -Path $sentinelFile | Out-Null
            }
            Write-Host ""
            Write-Host "Infra up. Starting uvicorn on host..." -ForegroundColor Cyan
            Write-Host ""
            dotenv -f .env -f .env.local run -- uvicorn app.endpoint.main:app --reload
        }

    } else {
        # Everything in Docker, override switches app to uvicorn
        if ($needsBuild) {
            docker compose up --build -d
        } else {
            docker compose up -d
        }

        if ($LASTEXITCODE -eq 0 -and $needsBuild) {
            New-Item -ItemType File -Path $sentinelFile | Out-Null
        }
    }

} else {
    # Everything in Docker (gunicorn, no override)
    if ($needsBuild) {
        docker compose -f docker-compose.yaml up --build -d
    } else {
        docker compose -f docker-compose.yaml up -d
    }

    if ($LASTEXITCODE -eq 0 -and $needsBuild) {
        New-Item -ItemType File -Path $sentinelFile | Out-Null
    }
}
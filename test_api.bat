@echo off
echo Testing API...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8002/' -Method GET -TimeoutSec 5; Write-Host 'Status:' $response.StatusCode; Write-Host 'Response:' $response.Content } catch { Write-Host 'Error:' $_.Exception.Message }"
echo.
echo Testing prediction...
powershell -Command "try { $body = '{\"query\":\"My computer won''''t start\"}'; $response = Invoke-WebRequest -Uri 'http://localhost:8002/predict' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 30; Write-Host 'Status:' $response.StatusCode; Write-Host 'Response:' $response.Content } catch { Write-Host 'Error:' $_.Exception.Message }"
pause
Write-Host "Configurando Variáveis de Ambiente do Ollama para Otimização de Performance" -ForegroundColor Cyan

# Variáveis a serem configuradas
$envVars = @{
    "OLLAMA_NUM_PARALLEL" = "4"
    "OLLAMA_MAX_LOADED_MODELS" = "2"
    "OLLAMA_FLASH_ATTENTION" = "1"
}

foreach ($key in $envVars.Keys) {
    $value = $envVars[$key]
    Write-Host "Configurando $key=$value"
    [Environment]::SetEnvironmentVariable($key, $value, "User")
}

Write-Host "`nConfigurações aplicadas com sucesso!" -ForegroundColor Green
Write-Host "Por favor, reinicie o Ollama (feche o ícone na bandeja do sistema e abra novamente) para que as alterações entrem em vigor." -ForegroundColor Yellow
Write-Host "Para aplicar essas variáveis no terminal atual imediatamente, feche e abra o terminal." -ForegroundColor Yellow

@echo off
setlocal

set GLSL_DIR=%~dp0glsl
set SPIRV_DIR=%~dp0spirv

if not exist "%SPIRV_DIR%" mkdir "%SPIRV_DIR%"

echo Compiling shaders...

glslc "%GLSL_DIR%\rgb_to_xyb.comp" -o "%SPIRV_DIR%\rgb_to_xyb.spv"
if errorlevel 1 (echo FAILED: rgb_to_xyb.comp & exit /b 1)
echo   rgb_to_xyb.comp OK

glslc "%GLSL_DIR%\gaussian_blur.comp" -o "%SPIRV_DIR%\gaussian_blur.spv"
if errorlevel 1 (echo FAILED: gaussian_blur.comp & exit /b 1)
echo   gaussian_blur.comp OK

glslc "%GLSL_DIR%\downsample.comp" -o "%SPIRV_DIR%\downsample.spv"
if errorlevel 1 (echo FAILED: downsample.comp & exit /b 1)
echo   downsample.comp OK

glslc "%GLSL_DIR%\ssim_error.comp" -o "%SPIRV_DIR%\ssim_error.spv"
if errorlevel 1 (echo FAILED: ssim_error.comp & exit /b 1)
echo   ssim_error.comp OK

echo.
echo All shaders compiled successfully!

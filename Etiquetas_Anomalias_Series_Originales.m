%Los 1 son VALORES NO ANOMALOS Y LOS 0 ANÓMALOS

% Leer el archivo CSV
datos = readtable('originales.csv');

% Obtener el encabezado (primera fila) del archivo CSV
encabezado = datos.Properties.VariableNames;

% Inicializar un vector para almacenar los resultados
resultado = zeros(1, numel(encabezado));

% Iterar sobre cada columna del encabezado
for i = 1:numel(encabezado)
    % Verificar si la columna contiene la cadena "NoAnomalo"
    if contains(encabezado{i}, 'NoAnomalo')
        % Si contiene la cadena, asignar 1 al resultado correspondiente
        resultado(i) = 1;
    end
end

% Mostrar el resultado
VectorAnomalias = resultado
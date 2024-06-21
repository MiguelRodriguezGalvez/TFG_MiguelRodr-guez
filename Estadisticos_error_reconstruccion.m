% Cargar datos originales
originales = readtable('originales.csv');

% Cargar datos reconstruidos
reconstruidos = readtable('reconstruidos.csv');

% Calcular el error promedio de reconstrucción para cada serie
error_promedio = mean(abs(table2array(originales) - table2array(reconstruidos)), 2);

% Calcular estadísticos del rendimiento de la reconstrucción
error_promedio_total = mean(error_promedio);
error_mediano_total = median(error_promedio);
error_maximo_total = max(error_promedio);
error_minimo_total = min(error_promedio);

fprintf('Estadísticos del rendimiento de la reconstrucción:\n');
fprintf('Error promedio: %.4f\n', error_promedio_total);
fprintf('Error mediano: %.4f\n', error_mediano_total);
fprintf('Error máximo: %.4f\n', error_maximo_total);
fprintf('Error mínimo: %.4f\n', error_minimo_total);

% Histograma del error de reconstrucción
figure;
histogram(error_promedio, 'BinWidth', 0.1);
title('Histograma del Error de Reconstrucción');
xlabel('Error Promedio');
ylabel('Frecuencia');






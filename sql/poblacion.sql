SELECT  
	fct.stridentificacion AS [ID], 
	fct.strcodestadoclientefuente, 
	dime.strDescEstadoFuente 
FROM BodegaCorporativa.[bodega].[factasociatividad] fct 
INNER JOIN BodegaCorporativa.bodega.dimestado dime
ON 
	fct.skEstadoCliente = dime.skEstado
	AND dime.numCodTipoEstado = 1
	AND dime.strFuente ='CooTaylor'
	AND dime.strDescEstadoFuente ='Inactivo'
WHERE
	BodegaCorporativa.$partition.Pf_mes(fct.dtmfechainsercion) =  BodegaCorporativa.$partition.pf_mes('?')
	AND fct.[indregistroactual] = 1 
	AND fct.strfuente = 'CooTaylor';
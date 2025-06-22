import numpy as np
import os

def read_asc_file(filepath):
    """Read an ESRI ASCII grid file and return header and data array."""
    header = {}
    with open(filepath, 'r') as f:
        for _ in range(6):
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1]) if line[0].lower() in ['xllcorner', 'yllcorner', 'cellsize', 'nodata_value'] else int(line[1])
        data = np.loadtxt(f, dtype=float).reshape(header['nrows'], header['ncols'])  # Ensure correct shape
        nodata_value = header.get('nodata_value', -9999)
        data[data == nodata_value] = np.nan
    print(f"Read {filepath} with shape: {data.shape}")
    return header, data

def get_mask(ncols, nrows, outlet_col, outlet_row, nodata_value, fdr, nextc, nextr, mask):
    """Generate a sub-basin mask starting from an outlet pixel."""
    sub_mask = np.zeros((ncols, nrows), dtype=int)
    sub_mask[outlet_col, outlet_row] = 1
    
    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(mask[j, i]) or mask[j, i] == 0:
                continue
            current_r, current_c = i, j
            while 0 <= current_r < nrows and 0 <= current_c < ncols:
                if nextc[current_c, current_r] == nodata_value or nextr[current_c, current_r] == nodata_value:
                    break
                next_r, next_c = nextr[current_c, current_r], nextc[current_c, current_r]
                if next_r == outlet_row and next_c == outlet_col:
                    sub_mask[j, i] = 1
                    break
                if sub_mask[next_c, next_r] == 1:
                    sub_mask[j, i] = 1
                    break
                current_r, current_c = next_r, next_c
    
    return sub_mask

def write_asc_file(filepath, data, header):
    """Write a grid to an ESRI ASCII file."""
    with open(filepath, 'w') as f:
        f.write(f"ncols {int(header['ncols'])}\n")
        f.write(f"nrows {int(header['nrows'])}\n")
        f.write(f"xllcorner {header['xllcorner']}\n")
        f.write(f"yllcorner {header['yllcorner']}\n")
        f.write(f"cellsize {header['cellsize']}\n")
        f.write(f"NODATA_value {header['nodata_value']}\n")
        data_out = np.where(np.isnan(data), header['nodata_value'], data)
        np.savetxt(f, data_out, fmt='%d', delimiter=' ')

def parallel_hydro_pre(mask_file, fac_file, stream_file, output_dir, n_subbasins, outlet_row, outlet_col, fdr_file=None):
    """Main function to process ASC files and generate sub-basins."""
    # Read input files
    mask_header, g_Mask = read_asc_file(mask_file)
    _, g_FAC = read_asc_file(fac_file)
    _, g_Stream = read_asc_file(stream_file)
    
    ncols, nrows = int(mask_header['ncols']), int(mask_header['nrows'])
    nodata_value = mask_header['nodata_value']
    
    # Ensure correct shape for all grids
    g_Mask = g_Mask.T if g_Mask.shape[0] != ncols else g_Mask  # Transpose if needed
    g_FAC = g_FAC.T if g_FAC.shape[0] != ncols else g_FAC
    g_Stream = g_Stream.T if g_Stream.shape[0] != ncols else g_Stream
    
    # Read flow direction file if provided
    if fdr_file:
        _, g_FDR = read_asc_file(fdr_file)
        g_FDR = g_FDR.T if g_FDR.shape[0] != ncols else g_FDR  # Transpose if needed
    else:
        raise ValueError("fdr_file is required to avoid index errors with the current implementation.")
    
    # Compute g_NextR and g_NextC from FDR
    g_NextR = np.full((ncols, nrows), nodata_value, dtype=int)
    g_NextC = np.full((ncols, nrows), nodata_value, dtype=int)
    directions = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    for i in range(nrows):
        for j in range(ncols):
            if not np.isnan(g_FDR[j, i]) and g_FDR[j, i] in directions:
                dr, dc = directions[g_FDR[j, i]]
                ni, nj = i + dr, j + dc
                if 0 <= ni < nrows and 0 <= nj < ncols:
                    g_NextR[j, i] = ni
                    g_NextC[j, i] = nj
    
    # Initialize arrays
    Npixel_channel = np.sum(g_Stream == 1)
    ChannelIndex_Rows = np.zeros(Npixel_channel, dtype=int)
    ChannelIndex_Cols = np.zeros(Npixel_channel, dtype=int)
    Channel_FAC = np.zeros(Npixel_channel, dtype=int)
    
    # Record channel cells
    channel_count = 0
    for i in range(nrows):
        for j in range(ncols):
            if g_Stream[j, i] == 1:
                ChannelIndex_Rows[channel_count] = i
                ChannelIndex_Cols[channel_count] = j
                Channel_FAC[channel_count] = g_FAC[j, i]
                channel_count += 1
    
    # Calculate next channel indices
    outlet_index = np.argmax(Channel_FAC)
    NextChannel = np.zeros(Npixel_channel, dtype=int)
    for i_channel_pixel in range(Npixel_channel):
        i_next = g_NextR[ChannelIndex_Cols[i_channel_pixel], ChannelIndex_Rows[i_channel_pixel]]
        j_next = g_NextC[ChannelIndex_Cols[i_channel_pixel], ChannelIndex_Rows[i_channel_pixel]]
        for j_channel_pixel in range(Npixel_channel):
            if i_next == ChannelIndex_Rows[j_channel_pixel] and j_next == ChannelIndex_Cols[j_channel_pixel]:
                NextChannel[i_channel_pixel] = j_channel_pixel
                break
    NextChannel[outlet_index] = outlet_index + 1
    
    # Create sub-basins
    FAC_divide = n_subbasins
    Residual_Channel_FAC = np.zeros(Npixel_channel, dtype=int)
    Channel_connect = np.zeros(n_subbasins, dtype=int)
    Subbasin_assemble = np.zeros((n_subbasins, ncols, nrows), dtype=int)
    g_FDR_forSub = np.copy(g_FDR)
    
    for i_basin in range(1, n_subbasins):
        i_basin_str = f"{i_basin:03d}"
        FAC_Subbasin_outlet = np.nanmax(Channel_FAC) // FAC_divide
        Residual_Channel_FAC = np.where(Channel_FAC > 0, Channel_FAC - FAC_Subbasin_outlet, 0)
        Residual_Channel_FAC = np.where(Residual_Channel_FAC < 0, -Residual_Channel_FAC, Residual_Channel_FAC)
        Residual_Channel_FAC = np.where(Residual_Channel_FAC == 0, -nodata_value, Residual_Channel_FAC)
        
        Index_Subbasin_outlet = np.argmin(Residual_Channel_FAC)
        FAC_Subbasin_outlet = Channel_FAC[Index_Subbasin_outlet]
        ii = ChannelIndex_Rows[Index_Subbasin_outlet]
        jj = ChannelIndex_Cols[Index_Subbasin_outlet]
        Channel_connect[i_basin - 1] = Index_Subbasin_outlet
        
        g_SubMask = get_mask(ncols, nrows, jj, ii, nodata_value, g_FDR_forSub, g_NextC, g_NextR, g_Mask)
        
        output_file = os.path.join(output_dir, f"Subbasin_#{i_basin_str}.asc")
        write_asc_file(output_file, g_SubMask, mask_header)
        
        for i_channel_pixel in range(Npixel_channel):
            NextChannel_index = NextChannel[i_channel_pixel]
            while NextChannel_index < Npixel_channel:
                if NextChannel_index == Index_Subbasin_outlet:
                    Channel_FAC[i_channel_pixel] = nodata_value
                    break
                NextChannel_index = NextChannel[NextChannel_index]
        
        Channel_FAC[Index_Subbasin_outlet] -= FAC_Subbasin_outlet
        NextChannel_index = NextChannel[Index_Subbasin_outlet]
        while NextChannel_index <= Npixel_channel:
            Channel_FAC[NextChannel_index] -= FAC_Subbasin_outlet
            NextChannel_index = NextChannel[NextChannel_index]
            if NextChannel_index == NextChannel[outlet_index]:
                break
        
        g_FDR_forSub = np.where(g_SubMask == 1, nodata_value, g_FDR_forSub)
        FAC_divide -= 1
        Residual_Channel_FAC = np.zeros(Npixel_channel, dtype=int)
        Subbasin_assemble[i_basin - 1, :, :] = g_SubMask
        print(f"  Generating Subbasin_#{i_basin_str}......done")
    
    # Last sub-basin using the outlet
    i_basin_str = f"{n_subbasins:03d}"
    Channel_connect[n_subbasins - 1] = outlet_index
    g_SubMask = get_mask(ncols, nrows, outlet_col, outlet_row, nodata_value, g_FDR_forSub, g_NextC, g_NextR, g_Mask)
    
    output_file = os.path.join(output_dir, f"Subbasin_#{i_basin_str}.asc")
    write_asc_file(output_file, g_SubMask, mask_header)
    Subbasin_assemble[n_subbasins - 1, :, :] = g_SubMask
    print(f"  Generating Subbasin_#{i_basin_str}......done")
    print("  All Subbasins have been generated.")
    
    return ChannelIndex_Rows, ChannelIndex_Cols, Channel_connect, Subbasin_assemble

# Example usage
if __name__ == "__main__":
    mask_file = "D:/Downloads/PHyL_v1.0-1.0/PHyL_v1.0-1.0/HydroBasics/mask.asc"
    fac_file = "D:/Downloads/PHyL_v1.0-1.0/PHyL_v1.0-1.0/HydroBasics/FAC.asc"
    stream_file = "D:/Downloads/PHyL_v1.0-1.0/PHyL_v1.0-1.0/HydroBasics/stream.asc"
    fdr_file = "D:/Downloads/PHyL_v1.0-1.0/PHyL_v1.0-1.0/HydroBasics/FDR.asc"
    output_dir = "D:/Tool"
    n_subbasins = 4
    outlet_row = 163
    outlet_col = 564

    os.makedirs(output_dir, exist_ok=True)
    ChannelIndex_Rows, ChannelIndex_Cols, Channel_connect, Subbasin_assemble = parallel_hydro_pre(
        mask_file, fac_file, stream_file, output_dir, n_subbasins, outlet_row, outlet_col, fdr_file
    )

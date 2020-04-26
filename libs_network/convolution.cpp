#include "convolution.h"
#include <stdint.h>


template<   unsigned int kernel_size, unsigned int input_channels, class io_data_type = int8_t, class acc_data_type = int16_t>
void conv2D_kernel(     io_data_type *output_buffer, 
                        io_data_type *input_buffer, 
                        io_data_type *kernel, 

                        unsigned int output_channels,
                        unsigned int height, 
                        unsigned int width,
                        unsigned int stride)
{
    unsigned int k_half         = (kernel_size - 1)/2;
    unsigned int input_size_y   = height    - 2*k_half;
    unsigned int input_size_x   = width     - 2*k_half;
    
    for (unsigned int filter = 0; filter < output_channels; filter++)
        for (unsigned int y = 0; y <= input_size_y-stride/2; y+= stride)
            for (unsigned int x = 0; x <= input_size_x-stride/2; x+= stride)
            {
                unsigned int kernel_idx = filter*kernel_size*kernel_size*input_channels;
 
                acc_data_type result = 0;
                 
                for (unsigned int ky = 0; ky < kernel_size; ky++)
                { 
                    unsigned int input_idx  = ((y + ky)*width + x)*input_channels;                   
                    
                    if (input_channels%32 == 0)
                    { 
                        for (unsigned int i = 0; i < kernel_size*input_channels; i+= 32)
                        {   
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                        }
                    }
                    else if (input_channels%16 == 0)
                    { 
                        for (unsigned int i = 0; i < kernel_size*input_channels; i+= 16)
                        {   
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                        }
                    }
                    else if (input_channels%8 == 0)
                    {
                        for (unsigned int i = 0; i < kernel_size*input_channels; i+= 8)
                        {
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                        } 
                    }
                    else if (input_channels%4 == 0)
                    {
                        for (unsigned int i = 0; i < kernel_size*input_channels; i+= 4)
                        {
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                            result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                        }
                    } 
                    else
                    {
                        result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                        result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                        result+= ((int16_t)kernel[kernel_idx])*((int16_t)input_buffer[input_idx]); kernel_idx++; input_idx++;
                    }
                }
                
                result = result/128;

                if (result > 127)
                    result = 127;
                
                if (result < -127)
                    result = -127;

                unsigned int x_output = x/stride + k_half;
                unsigned int y_output = y/stride + k_half;

                unsigned int output_idx     = ((y_output)*(width/stride) + x_output)*output_channels + filter;
                output_buffer[output_idx]   = result; 
            }
}



void conv2D(    int8_t *output_buffer, 
                int8_t *input_buffer, 
                int8_t *kernel, 

                unsigned int output_channels,
                unsigned int height, 
                unsigned int width,
                unsigned int input_channels,
                unsigned int stride)
{

    if (input_channels == 1) 
    {
        conv2D_kernel<3, 1, int8_t, int16_t>(   output_buffer, 
                                                input_buffer, 
                                                kernel, 

                                                output_channels,
                                                height, 
                                                width,

                                                stride);
    }
    else if (input_channels == 4) 
    {
        conv2D_kernel<3, 4, int8_t, int16_t>(   output_buffer, 
                                                input_buffer, 
                                                kernel, 

                                                output_channels,
                                                height, 
                                                width,

                                                stride);
    }
    else if (input_channels == 8)
    {
        conv2D_kernel<3, 8, int8_t, int16_t>(   output_buffer, 
                                                input_buffer, 
                                                kernel, 

                                                output_channels,
                                                height, 
                                                width,
                                                
                                                stride);
    }
    else if (input_channels == 16)
    {
        conv2D_kernel<3, 16, int8_t, int16_t>(  output_buffer, 
                                                input_buffer, 
                                                kernel, 

                                                output_channels,
                                                height, 
                                                width,
                                                
                                                stride);
    }
    else if (input_channels == 32)
    {
        conv2D_kernel<3, 32, int8_t, int16_t>(  output_buffer, 
                                                input_buffer, 
                                                kernel, 

                                                output_channels,
                                                height, 
                                                width,
                                                
                                                stride);
    }
    else if (input_channels == 64)
    {
        conv2D_kernel<3, 64, int8_t, int16_t>(  output_buffer, 
                                                input_buffer, 
                                                kernel, 

                                                output_channels,
                                                height, 
                                                width,
                                                
                                                stride);
    }
}
#include "EagleeyeModule.h"

extern "C"{
    /**
     * @brief initialize ${project} pipeline
     * 
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_initialize(const char* config_folder);

    /**
     * @brief load ${project} pipeline configure
     * 
     * @param folder 
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_load_config(const char* config_file);

    /**
     * @brief save ${project} pipeline configure
     * 
     * @param folder 
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_save_config(const char* config_file);

    /**
     * @brief release ${project} pipeline
     * 
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_release();

    /**
     * @brief run ${project} pipeline
     * 
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_run(const char* node_name=NULL, const char* ignore_prefix=NULL);

    /**
     * @brief get ${project} pipeline version
     * 
     * @param pipeline_version
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_version(char* pipeline_version);

    /**
     * @brief reset ${project} pipeline state
     * 
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_reset();

    /**
     * @brief set any node param in ${project} pipeline
     * 
     * @param node_name node name in pipeline
     * @param param_name node param in pipeline
     * @param value param value
     * @return true  success to set
     * @return false fail to set
     */
    bool eagleeye_${project}_set_param(const char* node_name, const char* param_name, const void* value);

    /**
     * @brief get any node param in ${project} pipeline
     * 
     * @param node_name node name in pipeline
     * @param param_name node param in pipeline
     * @param value param value
     * @return true success to get
     * @return false fail to get
     */
    bool eagleeye_${project}_get_param(const char* node_name, const char* param_name, void* value);

    /**
     * @brief set input data from ${project} input node
     * 
     * @param node_name node name in pipeline
     * @param data data pointer
     * @param data_size dimension (H,W,C)
     * @param data_dims dimension number
     * @param data_type data type
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_set_input(const char* node_name, void* data, const size_t* data_size, const int data_dims, const int data_rotation, const int data_type);
    bool eagleeye_${project}_set_input2(const char* node_name, void* data, EagleeyeMeta meta);

    /**
     * @brief get output data from ${project} output node
     * 
     * @param node_name node name in pipeline
     * @param data data pointer
     * @param data_size dimension (H,W,C)/(B,H,W,C)
     * @param data_dims dimension number
     * @param data_type data type
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_get_output(const char* node_name, void*& data, size_t*& data_size, int& data_dims,int& data_type);

    /**
     * @brief replace node/port with placeholder(debug)
     * 
     * @param node_name node name in pipeline
     * @param port port
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_debug_replace_at(const char* node_name, int port);

    /**
     * @brief restore node/port(debug)
     * 
     * @param node_name node name in pipeline
     * @param port port
     * @return true 
     * @return false 
     */
    bool eagleeye_${project}_debug_restore_at(const char* node_name, int port);
}
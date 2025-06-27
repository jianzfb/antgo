// >>>>>>>>>>>>>>>>>>>>>>AUTOGENERATE PLUGIN HEADER>>>>>>>>>>>>>>>>>>>>>>
#include "eagleeye/common/EagleeyeMacro.h"
#include "eagleeye/framework/pipeline/AnyNode.h"
#include "eagleeye/framework/pipeline/SignalFactory.h"
#include "eagleeye/framework/pipeline/AnyPipeline.h"
#include "eagleeye/processnode/Placeholder.h"
${include_list}
#include "${project}_plugin.h"
#include "eagleeye/processnode/NNNode.h"
#include "eagleeye/framework/pipeline/DynamicPipelineCreater.h"
#include "eagleeye/processnode/LambdaNode.h"
#include "eagleeye/framework/pipeline/JsonSignal.h"
#include "eagleeye/processnode/AutoNode.h"
#include "eagleeye/processnode/ProxyNode.h"
// >>>>>>>>>>>>>>>>>>>>>>AUTOGENERATE PLUGIN HEADER>>>>>>>>>>>>>>>>>>>>>>

// DEVELOPER WRITE HEADER HERE

namespace eagleeye{
using namespace dataflow;

/**
 * @brief register pipeline plugin
 * 
 */
EAGLEEYE_PIPELINE_REGISTER(${project}, ${version}, ${signature});

/**
 * @brief configure pipeline plugin
 * 
 */
EAGLEEYE_BEGIN_PIPELINE_INITIALIZE(${project})
// >>>>>>>>>>>>>>>>>>>>>>AUTOGENERATE PLUGIN SOURCE>>>>>>>>>>>>>>>>>>>>>>
// 1.step build datasource node
std::vector<int> in_port_list = std::vector<int>(${in_port});
std::vector<std::string> in_signal_list = std::vector<std::string>(${in_signal});
for(size_t i=0; i<in_port_list.size(); ++i){
    AnyNode* input_node = NULL;
    std::string type_str = in_signal_list[i];
    if(type_str == "EAGLEEYE_SIGNAL_IMAGE" ||
        type_str == "EAGLEEYE_SIGNAL_RGB_IMAGE" ||
        type_str == "EAGLEEYE_SIGNAL_BGR_IMAGE"){
        input_node = new Placeholder<ImageSignal<Array<unsigned char, 3>>>(${is_asyn}, 10, false, true);        
        if(type_str == "EAGLEEYE_SIGNAL_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_IMAGE);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_RGB_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_RGB_IMAGE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_BGR_IMAGE);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_RGBA_IMAGE" ||
            type_str == "EAGLEEYE_SIGNAL_BGRA_IMAGE"){
        input_node = new Placeholder<ImageSignal<Array<unsigned char, 4>>>(${is_asyn}, 10, false, true);

        if(type_str == "EAGLEEYE_SIGNAL_RGBA_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_RGBA_IMAGE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_BGRA_IMAGE);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_GRAY_IMAGE" ||
            type_str == "EAGLEEYE_SIGNAL_MASK" || 
            type_str == "EAGLEEYE_SIGNAL_UCMATRIX"){
        input_node = new Placeholder<ImageSignal<unsigned char>>(${is_asyn}, 10, false, true);

        if(type_str == "EAGLEEYE_SIGNAL_GRAY_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_GRAY_IMAGE);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_MASK"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_MASK);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_UCMATRIX);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_TENSOR"){
        input_node = new Placeholder<TensorSignal>(${is_asyn}, 10, false, true);        
        input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_TENSOR);
    }
    else if(type_str == "EAGLEEYE_SIGNAL_DET" ||
            type_str == "EAGLEEYE_SIGNAL_DET_EXT" ||
            type_str == "EAGLEEYE_SIGNAL_TRACKING" ||
            type_str == "EAGLEEYE_SIGNAL_POS_2D" ||
            type_str == "EAGLEEYE_SIGNAL_POS_3D" ||
            type_str == "EAGLEEYE_SIGNAL_LANDMARK" || 
            type_str == "EAGLEEYE_SIGNAL_FMATRIX"){
        input_node = new Placeholder<ImageSignal<float>>(${is_asyn}, 10, false, true);
        if(type_str == "EAGLEEYE_SIGNAL_DET"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_DET);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_DET_EXT"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_DET_EXT);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_TRACKING"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_TRACKING);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_POS_2D"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_POS_2D);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_POS_3D"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_POS_3D);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_LANDMARK"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_LANDMARK);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_FMATRIX);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_CLS" ||
            type_str == "EAGLEEYE_SIGNAL_STATE" ||
            type_str == "EAGLEEYE_SIGNAL_IMATRIX"){
        input_node = new Placeholder<ImageSignal<int>>(${is_asyn}, 10, false, true);
        if(type_str == "EAGLEEYE_SIGNAL_CLS"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_CLS);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_STATE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_STATE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_IMATRIX);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_SWITCH"){
        input_node = new Placeholder<BooleanSignal>(${is_asyn}, 10, false, true);
        input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_SWITCH);
    }
    else if(type_str == "EAGLEEYE_SIGNAL_RECT" ||
            type_str == "EAGLEEYE_SIGNAL_LINE" ||
            type_str == "EAGLEEYE_SIGNAL_POINT"){
        input_node = new Placeholder<ImageSignal<float>>(${is_asyn}, 10, false, true);
        if(type_str == "EAGLEEYE_SIGNAL_RECT"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_RECT);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_LINE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_LINE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_POINT);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_TIMESTAMP"){
        input_node = new Placeholder<ImageSignal<double>>(${is_asyn}, 10, false, true);
        input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_TIMESTAMP);
    }
    else{
        EAGLEEYE_LOGE("Dont support signal type %s.", type_str.c_str());
    }

    std::string placeholer_i_name = formatString("placeholder_%d", i);
    ${project}->add(input_node, placeholer_i_name.c_str());
}

// 2.step build your algorithm node
std::vector<AnyNode*> nnnodes = {
    ${node_graph}
};

// 3.step add all node to pipeline
// 3.1.step add data source node
// 3.2.step add your algorithm node
std::vector<std::string> nnnames = ${nnnames};
for(int i=0; i<nnnodes.size(); ++i){
    ${project}->add(nnnodes[i], nnnames[i].c_str());
}

// 4.step link all node in pipeline
// in link
std::vector<std::vector<std::pair<std::string, int>>> in_links = ${in_links};
for(int i=0; i<in_links.size(); ++i){
    std::string placeholer_i_name = formatString("placeholder_%d", i);
    for(int in_i=0; in_i<in_links[i].size(); ++in_i){
        ${project}->bind(placeholer_i_name.c_str(), 0, in_links[i][in_i].first.c_str(), in_links[i][in_i].second);
    }
}

// out link
std::vector<std::pair<std::string, int>> out_links = ${out_links};


// between link
std::vector<std::pair<std::string, int>> from_links = ${from_links};
std::vector<std::pair<std::string, int>> to_links = ${to_links};
for(int i=0; i<from_links.size(); ++i){
    ${project}->bind(from_links[i].first.c_str(), from_links[i].second, to_links[i].first.c_str(), to_links[i].second);
}

// >>>>>>>>>>>>>>>>>>>>>>AUTOGENERATE PLUGIN SOURCE>>>>>>>>>>>>>>>>>>>>>>

// DEVELOPER WRITE SOURCE HERE

EAGLEEYE_END_PIPELINE_INITIALIZE(${project})
}
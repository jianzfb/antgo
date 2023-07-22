ANTGO_API void* ${func_idcode_hash}_init(${init_args_def}) {
  ${class_name}* obj_ptr = new ${class_name}(${init_args_inst});
  return (void*)obj_ptr;
}

ANTGO_API void ${func_idcode_hash}_run(void* void_ptr, ${run_args_def}) {
  ${class_name}* obj_ptr = (${class_name}*)void_ptr;
  obj_ptr->run(${run_args_inst});
}

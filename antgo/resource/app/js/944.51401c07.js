"use strict";(self["webpackChunkantgo_web"]=self["webpackChunkantgo_web"]||[]).push([[944],{9944:function(t,e,l){l.r(e),l.d(e,{default:function(){return r}});var i=function(){var t=this,e=t._self._c;return e("div",[e("b-container",[e("div",[e("b-row",{attrs:{"align-h":"center"}},[e("b-col",[e("b-dropdown",{staticClass:"m-md-2",staticStyle:{width:"150px"},attrs:{text:"过滤",variant:"outline-primary"}},t._l(t.columns,(function(l){return e("b-dropdown-item",{key:l},[t._v(" "+t._s(l)+" ")])})),1)],1),e("b-col",{staticStyle:{display:"flex","flex-direction":"row","justify-content":"center","align-content":"center","flex-wrap":"wrap"}},[e("div",{staticClass:"m-md-2"},[e("b-badge",{staticStyle:{height:"30px",display:"flex","flex-direction":"column","justify-content":"center"},attrs:{pill:"",variant:"info"}},[t._v(" "+t._s(t.sample_num_completed)+"/"+t._s(t.sample_num)+" ")])],1)]),e("b-col",[e("div",{staticStyle:{float:"right"}},[e("b-button",{staticClass:"m-md-2",staticStyle:{width:"100px"},attrs:{variant:"outline-primary"},on:{click:function(e){return t.showUploadWindow()}}},[t._v("导入")]),e("b-button",{staticClass:"m-md-2",staticStyle:{width:"100px"},attrs:{variant:"outline-primary"},on:{click:function(e){return t.export_samples()}}},[t._v("导出")]),e("b-button",{staticClass:"m-md-2",staticStyle:{width:"100px"},attrs:{variant:"danger"},on:{click:function(e){return t.finish_label()}}},[t._v("完成")])],1)])],1),e("b-row",{staticStyle:{height:"70px",width:"auto"}},[e("b-col",[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"40px",display:"flex",height:"100%","align-items":"center"}},[e("b-form-checkbox",{model:{value:t.select_all,callback:function(e){t.select_all=e},expression:"select_all"}})],1)])]),t._l(t.heads,(function(l){return e("b-col",{key:l},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(l)+" ")])])])}))],2),t._l(t.samples,(function(l,i){return e("b-row",{key:i,staticClass:"row_activate",staticStyle:{cursor:"pointer"}},[e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"40px",display:"flex",height:"100%","align-items":"center"}},[e("b-form-checkbox",{model:{value:l["selected"],callback:function(e){t.$set(l,"selected",e)},expression:"sample['selected']"}})],1)])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(l,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(i)+" ")])])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(l,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},["completed"==l["state"]?e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(l["completed_time"])+" ")]):t._e()])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(l,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(l["label_info"].length)+" ")])])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(l,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},t._l(l["operator"],(function(l,i){return e("div",{key:i,staticClass:"lsf-space lsf-space_direction_horizontal lsf-space_size_small"},[e("div",{staticClass:"lsf-userpic lsf-annotations-list__userpic",staticStyle:{background:"rgb(179, 218, 216)",color:"rgb(0, 0, 0)"},attrs:{block:"lsf-annotations-list"}},[e("img",{staticClass:"lsf-userpic__avatar",staticStyle:{opacity:"0"},attrs:{alt:l["full_name"]}}),e("span",{staticClass:"lsf-userpic__username"},[t._v(t._s(l["short_name"]))])])])})),0)])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(l,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("img",{attrs:{src:l["image_file"],width:"40",height:"40"}})])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("b-icon",{staticStyle:{cursor:"pointer"},attrs:{icon:"code-slash",scale:"1",variant:"info"}})],1)])])],1)}))],2)]),e("input",{ref:"file_control",staticStyle:{display:"none"},attrs:{type:"file"},on:{change:t.selectedFile}})],1)},n=[],a={name:"Project",data(){return{columns:[],heads:["ID","完成时间","标注数","人员","图像","信息"],samples:[],select_all:!1,now_page:0,more_show:!0,sample_num_completed:0,sample_num:0}},mounted:function(){var t=this;this.axios.get("/antgo/api/info/").then((function(e){var l=e.data.content["project_state"]["stage"];"labeling"!=l?t.$router.push({path:"/"}):(t.sample_num_completed=e.data.content["project_state"]["sample_num_completed"],t.sample_num=e.data.content["project_state"]["sample_num"],t.scrollMore())})).catch((function(t){})),window.addEventListener("scroll",(()=>{const t=document.documentElement.scrollTop||document.body.scrollTop,e="CSS1Compat"===document.compatMode?document.documentElement.clientHeight:document.body.clientHeight,l=Math.max(document.body.scrollHeight,document.documentElement.scrollHeight);t+e>=l&&this.scrollMore()}))},methods:{enter:function(t,e){this.$router.push({path:"/project/"+e})},scrollMore(){var t=this;t.axios.get("/antgo/api/label/sample/",{params:{page_index:t.now_page,page_size:50}}).then((function(e){var l=e.data.content["total_sample_num"];for(var i in e.data.content["page_samples"])t.samples.push(e.data.content["page_samples"][i]);t.samples.length<l?t.more_show=!0:t.more_show=!1,t.now_page+=1})).catch((function(t){}))},export_samples:function(){this.axios({method:"get",url:"/antgo/api/label/export/",responseType:"blob"}).then((t=>{this.download(t)})).catch((t=>{}))},download:function(t){if(!t)return;let e=t.headers["content-disposition"].split(";")[1].split("=")[1],l=window.URL.createObjectURL(t["data"]),i=document.createElement("a");i.style.display="none",i.href=l,i.setAttribute("download",e),document.body.appendChild(i),i.click()},selectedFile:function(){let t=this.$refs.file_control.files[0];var e=t.name;let l=e.lastIndexOf("."),i=e.substring(l+1);"json"==i&&alert(e)},showUploadWindow:function(){console.log("hello"),this.$refs.file_control.dispatchEvent(new MouseEvent("click"))},finish_label:function(){var t=this;let e=new FormData;e.append("running_state","running"),e.append("running_stage","finish"),t.axios.post("/antgo/api/info/",e).then((function(e){t.$router.push({path:"/"})})).catch((function(t){console.log(t)}))}}},s=a,c=l(1001),o=(0,c.Z)(s,i,n,!1,null,"125ff328",null),r=o.exports}}]);
//# sourceMappingURL=944.51401c07.js.map
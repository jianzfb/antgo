"use strict";(self["webpackChunkantgo_web"]=self["webpackChunkantgo_web"]||[]).push([[854],{5854:function(t,e,n){n.r(e),n.d(e,{default:function(){return r}});var i=function(){var t=this,e=t._self._c;return e("div",[e("b-container",[e("div",[e("b-row",{attrs:{"align-h":"center"}},[e("b-col",[e("b-dropdown",{staticClass:"m-md-2",staticStyle:{width:"150px"},attrs:{text:"过滤",variant:"outline-primary"}},t._l(t.columns,(function(n){return e("b-dropdown-item",{key:n},[t._v(" "+t._s(n)+" ")])})),1)],1),e("b-col",{staticStyle:{display:"flex","flex-direction":"row","justify-content":"center","align-content":"center","flex-wrap":"wrap"}},[e("div",{staticClass:"m-md-2"},[e("b-badge",{staticStyle:{height:"30px",display:"flex","flex-direction":"column","justify-content":"center"},attrs:{pill:"",variant:"info"}},[t._v(" "+t._s(t.sample_num_completed)+"/"+t._s(t.sample_num)+" ")])],1)]),e("b-col",[e("div",{staticStyle:{float:"right"}},[e("b-button",{staticClass:"m-md-2",staticStyle:{width:"100px"},attrs:{variant:"outline-primary"},on:{click:function(e){return t.showUploadWindow()}}},[t._v("导入")]),e("b-button",{staticClass:"m-md-2",staticStyle:{width:"100px"},attrs:{variant:"outline-primary"},on:{click:function(e){return t.export_samples()}}},[t._v("导出")]),e("b-button",{staticClass:"m-md-2",staticStyle:{width:"100px"},attrs:{variant:"danger"},on:{click:function(e){return t.finish_label()}}},[t._v("完成")])],1)])],1),e("b-row",{staticStyle:{height:"70px",width:"auto"}},[e("b-col",[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"40px",display:"flex",height:"100%","align-items":"center"}},[e("b-form-checkbox",{model:{value:t.select_all,callback:function(e){t.select_all=e},expression:"select_all"}})],1)])]),t._l(t.heads,(function(n){return e("b-col",{key:n},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(n)+" ")])])])}))],2),t._l(t.samples,(function(n,i){return e("b-row",{key:i,staticClass:"row_activate",staticStyle:{cursor:"pointer"}},[e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"40px",display:"flex",height:"100%","align-items":"center"}},[e("b-form-checkbox",{model:{value:n["selected"],callback:function(e){t.$set(n,"selected",e)},expression:"sample['selected']"}})],1)])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(n,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(i)+" ")])])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(n,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},["completed"==n["state"]?e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(n["completed_time"])+" ")]):t._e()])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(n,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},[t._v(" "+t._s(n["label_info"].length)+" ")])])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(n,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("div",{staticStyle:{width:"180px","justify-content":"space-between",display:"flex",height:"100%","align-items":"center"}},t._l(n["operator"],(function(n,i){return e("div",{key:i,staticClass:"lsf-space lsf-space_direction_horizontal lsf-space_size_small"},[e("div",{staticClass:"lsf-userpic lsf-annotations-list__userpic",staticStyle:{background:"rgb(179, 218, 216)",color:"rgb(0, 0, 0)"},attrs:{block:"lsf-annotations-list"}},[e("img",{staticClass:"lsf-userpic__avatar",staticStyle:{opacity:"0"},attrs:{alt:n["full_name"]}}),e("span",{staticClass:"lsf-userpic__username"},[t._v(t._s(n["short_name"]))])])])})),0)])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"},on:{click:function(e){return t.enter(n,i)}}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("img",{attrs:{src:n["image_file"],width:"40",height:"40"}})])])]),e("b-col",{staticStyle:{display:"flex","flex-direction":"column","flex-wrap":"nowrap","justify-content":"center"}},[e("div",{staticClass:"dm-table__cell"},[e("div",{staticStyle:{width:"110px",display:"flex",height:"100%","align-items":"center"}},[e("b-icon",{staticStyle:{cursor:"pointer"},attrs:{icon:"code-slash",scale:"1",variant:"info"}})],1)])])],1)}))],2)]),e("input",{ref:"file_control",staticStyle:{display:"none"},attrs:{type:"file"},on:{change:t.selectedFile}}),e("div",[e("b-modal",{attrs:{id:"finish_message",title:"提示信息","ok-title":"确认","cancel-title":"取消"},on:{ok:t.force_finish_label}},[e("p",{staticClass:"my-4"},[t._v("未完成所有样本标注, 是否强制结束？")])])],1)],1)},a=[],l=(n(3251),n(9452),n(4133),n(5053),{name:"Project",data(){return{columns:[],heads:["ID","完成时间","标注数","人员","图像","信息"],samples:[],select_all:!1,now_page:0,more_show:!0,sample_num_completed:0,sample_num:0}},mounted:function(){var t=this;this.axios.get("/antgo/api/info/").then((function(e){var n=e.data.content["project_state"]["stage"];"labeling"!=n?t.$router.push({path:"/"}):(t.sample_num_completed=e.data.content["project_state"]["sample_num_completed"],t.sample_num=e.data.content["project_state"]["sample_num"],t.scrollMore())})).catch((function(t){})),window.addEventListener("scroll",(()=>{const t=document.documentElement.scrollTop||document.body.scrollTop,e="CSS1Compat"===document.compatMode?document.documentElement.clientHeight:document.body.clientHeight,n=Math.max(document.body.scrollHeight,document.documentElement.scrollHeight);t+e>=n&&this.scrollMore()}))},methods:{enter:function(t,e){this.$router.push({path:"/project/"+e})},scrollMore(){var t=this;t.axios.get("/antgo/api/label/sample/",{params:{page_index:t.now_page,page_size:50}}).then((function(e){var n=e.data.content["total_sample_num"];for(var i in e.data.content["page_samples"])t.samples.push(e.data.content["page_samples"][i]);t.samples.length<n?t.more_show=!0:t.more_show=!1,t.now_page+=1})).catch((function(t){}))},export_samples:function(){this.axios({method:"get",url:"/antgo/api/label/export/",responseType:"blob"}).then((t=>{this.download(t)})).catch((t=>{}))},download:function(t){if(!t)return;let e=t.headers["content-disposition"].split(";")[1].split("=")[1],n=window.URL.createObjectURL(t["data"]),i=document.createElement("a");i.style.display="none",i.href=n,i.setAttribute("download",e),document.body.appendChild(i),i.click()},selectedFile:function(){let t=this.$refs.file_control.files[0];var e=t.name;let n=e.lastIndexOf(".");e.substring(n+1)},showUploadWindow:function(){console.log("hello"),this.$refs.file_control.dispatchEvent(new MouseEvent("click"))},finish_label:function(){var t=this;let e=new FormData;e.append("running_state","running"),e.append("running_stage","finish"),t.axios.post("/antgo/api/info/",e).then((function(e){e.data.content["finish"]?t.$router.push({path:"/"}):t.$bvModal.show("finish_message")})).catch((function(t){console.log(t)}))},force_finish_label:function(){var t=this;let e=new FormData;e.append("running_state","running"),e.append("running_stage","finish"),e.append("force",!0),t.axios.post("/antgo/api/info/",e).then((function(e){t.$router.push({path:"/"})})).catch((function(t){console.log(t)}))}}}),s=l,c=n(4249),o=(0,c.Z)(s,i,a,!1,null,"500d50b3",null),r=o.exports},9452:function(t,e,n){var i=n(6166),a=n(3974),l=n(3428),s=n(4425),c=URLSearchParams,o=c.prototype,r=a(o.append),d=a(o["delete"]),p=a(o.forEach),u=a([].push),f=new c("a=1&a=2&b=3");f["delete"]("a",1),f["delete"]("b",void 0),f+""!=="a=2"&&i(o,"delete",(function(t){var e=arguments.length,n=e<2?void 0:arguments[1];if(e&&void 0===n)return d(this,t);var i=[];p(this,(function(t,e){u(i,{key:e,value:t})})),s(e,1);var a,c=l(t),o=l(n),f=0,h=0,m=!1,_=i.length;while(f<_)a=i[f++],m||a.key===c?(m=!0,d(this,a.key)):h++;while(h<_)a=i[h++],a.key===c&&a.value===o||r(this,a.key,a.value)}),{enumerable:!0,unsafe:!0})},4133:function(t,e,n){var i=n(6166),a=n(3974),l=n(3428),s=n(4425),c=URLSearchParams,o=c.prototype,r=a(o.getAll),d=a(o.has),p=new c("a=1");!p.has("a",2)&&p.has("a",void 0)||i(o,"has",(function(t){var e=arguments.length,n=e<2?void 0:arguments[1];if(e&&void 0===n)return d(this,t);var i=r(this,t);s(e,1);var a=l(n),c=0;while(c<i.length)if(i[c++]===a)return!0;return!1}),{enumerable:!0,unsafe:!0})},5053:function(t,e,n){var i=n(3237),a=n(3974),l=n(1213),s=URLSearchParams.prototype,c=a(s.forEach);i&&!("size"in s)&&l(s,"size",{get:function(){var t=0;return c(this,(function(){t++})),t},configurable:!0,enumerable:!0})}}]);
//# sourceMappingURL=854.e9d0dc32.js.map
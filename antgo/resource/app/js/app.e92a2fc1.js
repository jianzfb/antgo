(function(){"use strict";var t={839:function(t,e,n){var r=n(6369),o=function(){var t=this,e=t._self._c;return e("div",{attrs:{id:"app"}},[e("Navbar"),e("router-view")],1)},a=[],i=function(){var t=this,e=t._self._c;return e("div",[e("b-navbar",{attrs:{toggleable:"md",type:"dark",variant:"info"}},[e("b-navbar-toggle",{attrs:{target:"nav_collapse"}}),e("b-navbar-brand",{attrs:{href:"#"}},[t._v("ANTGO "+t._s(t.project_type))]),e("b-collapse",{attrs:{"is-nav":"",id:"nav_collapse"}},[e("b-navbar-nav",["LABEL"==t.project_type?e("b-nav-item",{attrs:{href:"/#/projects/"}},[t._v("Project /")]):t._e(),"BROWSER"==t.project_type?e("b-nav-item",{attrs:{href:"/#/browser/"}},[t._v("Project /")]):t._e(),"PREDICT"==t.project_type?e("b-nav-item",{attrs:{href:"/#/predict/"}},[t._v("Project /")]):t._e(),e("b-nav-item",{attrs:{href:"#",disabled:""}},[t._v(t._s(t.project_name))])],1),e("b-navbar-nav",{staticClass:"ml-auto"},[e("b-navbar-nav",[e("b-button",{on:{click:function(e){return t.show_user_info()}}},[t._v(t._s(t.short_name))])],1)],1)],1)],1),e("b-card",{directives:[{name:"show",rawName:"v-show",value:t.is_show_info,expression:"is_show_info"}],attrs:{title:"用户信息","sub-title":""}},[e("b-card-text",[t._v(" "+t._s(t.full_name)+" ")]),e("b-card-text",[t._v(" "+t._s(t.statistic_info)+" ")]),e("b-button",{attrs:{variant:"primary"},on:{click:t.close}},[t._v("关闭")])],1)],1)},u=[],c={name:"Navbar",data(){return{project_name:"",project_type:"",short_name:"",full_name:"",statistic_info:"",is_show_info:!1}},mounted:function(){var t=this;this.axios.get("/antgo/api/user/info/").then((function(e){t.project_name=e.data.content["task_name"],t.project_type=e.data.content["project_type"],t.short_name=e.data.content["short_name"],t.full_name=e.data.content["full_name"]})).catch((function(e){t.$router.push({path:"/Login/"})}))},methods:{show_user_info:function(){var t=this;this.axios.get("/antgo/api/user/info/").then((function(e){var n=e.data.content["statistic_info"];t.statistic_info=n,t.is_show_info=!0})).catch((function(t){}))},close:function(){this.is_show_info=!1}}},s=c,f=n(1001),l=(0,f.Z)(s,i,u,!1,null,null,null),d=l.exports,p={name:"navbar",components:{Navbar:d}},h=p,v=(0,f.Z)(h,o,a,!1,null,null,null),m=v.exports,b=n(2631);const _=()=>n.e(191).then(n.bind(n,8191)),g=()=>n.e(359).then(n.bind(n,359)),y=()=>n.e(769).then(n.bind(n,1769)),w=()=>n.e(651).then(n.bind(n,9651)),j=()=>n.e(761).then(n.bind(n,3761)),k=()=>n.e(854).then(n.bind(n,6854)),C=()=>n.e(880).then(n.bind(n,5880));r["default"].use(b.Z);var O=new b.Z({routes:[{path:"/",name:"Welcome",component:w},{path:"/projects",name:"Project",component:_},{path:"/project/:id?",name:"Label",component:g},{path:"/login",name:"Login",component:y},{path:"/browser",name:"Browser",component:j},{path:"/predict",name:"Predict",component:k},{path:"/demo",name:"Demo",component:C}]}),x=n(6265),E=n.n(x),P=n(5996),T=n(9425);n(7024);E().defaults.withCredentials=!0,E().defaults.headers.post["Content-Type"]="application/x-www-form-urlencoded",r["default"].prototype.axios=E(),r["default"].use(P.XG7),r["default"].use(T.A7),r["default"].config.productionTip=!1,new r["default"]({render:t=>t(m),router:O}).$mount("#app")}},e={};function n(r){var o=e[r];if(void 0!==o)return o.exports;var a=e[r]={exports:{}};return t[r].call(a.exports,a,a.exports,n),a.exports}n.m=t,function(){var t=[];n.O=function(e,r,o,a){if(!r){var i=1/0;for(f=0;f<t.length;f++){r=t[f][0],o=t[f][1],a=t[f][2];for(var u=!0,c=0;c<r.length;c++)(!1&a||i>=a)&&Object.keys(n.O).every((function(t){return n.O[t](r[c])}))?r.splice(c--,1):(u=!1,a<i&&(i=a));if(u){t.splice(f--,1);var s=o();void 0!==s&&(e=s)}}return e}a=a||0;for(var f=t.length;f>0&&t[f-1][2]>a;f--)t[f]=t[f-1];t[f]=[r,o,a]}}(),function(){n.n=function(t){var e=t&&t.__esModule?function(){return t["default"]}:function(){return t};return n.d(e,{a:e}),e}}(),function(){n.d=function(t,e){for(var r in e)n.o(e,r)&&!n.o(t,r)&&Object.defineProperty(t,r,{enumerable:!0,get:e[r]})}}(),function(){n.f={},n.e=function(t){return Promise.all(Object.keys(n.f).reduce((function(e,r){return n.f[r](t,e),e}),[]))}}(),function(){n.u=function(t){return"js/"+t+"."+{191:"0cf192b3",359:"afb86915",651:"c5601578",761:"f73c7c45",769:"b7247054",854:"d3c0e54f",880:"cba02e88"}[t]+".js"}}(),function(){n.miniCssF=function(t){return"css/"+t+"."+{191:"e004636f",359:"32c5c11e",854:"9e012a59"}[t]+".css"}}(),function(){n.g=function(){if("object"===typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(t){if("object"===typeof window)return window}}()}(),function(){n.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)}}(),function(){var t={},e="antgo-web:";n.l=function(r,o,a,i){if(t[r])t[r].push(o);else{var u,c;if(void 0!==a)for(var s=document.getElementsByTagName("script"),f=0;f<s.length;f++){var l=s[f];if(l.getAttribute("src")==r||l.getAttribute("data-webpack")==e+a){u=l;break}}u||(c=!0,u=document.createElement("script"),u.charset="utf-8",u.timeout=120,n.nc&&u.setAttribute("nonce",n.nc),u.setAttribute("data-webpack",e+a),u.src=r),t[r]=[o];var d=function(e,n){u.onerror=u.onload=null,clearTimeout(p);var o=t[r];if(delete t[r],u.parentNode&&u.parentNode.removeChild(u),o&&o.forEach((function(t){return t(n)})),e)return e(n)},p=setTimeout(d.bind(null,void 0,{type:"timeout",target:u}),12e4);u.onerror=d.bind(null,u.onerror),u.onload=d.bind(null,u.onload),c&&document.head.appendChild(u)}}}(),function(){n.r=function(t){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})}}(),function(){n.p="/"}(),function(){var t=function(t,e,n,r){var o=document.createElement("link");o.rel="stylesheet",o.type="text/css";var a=function(a){if(o.onerror=o.onload=null,"load"===a.type)n();else{var i=a&&("load"===a.type?"missing":a.type),u=a&&a.target&&a.target.href||e,c=new Error("Loading CSS chunk "+t+" failed.\n("+u+")");c.code="CSS_CHUNK_LOAD_FAILED",c.type=i,c.request=u,o.parentNode.removeChild(o),r(c)}};return o.onerror=o.onload=a,o.href=e,document.head.appendChild(o),o},e=function(t,e){for(var n=document.getElementsByTagName("link"),r=0;r<n.length;r++){var o=n[r],a=o.getAttribute("data-href")||o.getAttribute("href");if("stylesheet"===o.rel&&(a===t||a===e))return o}var i=document.getElementsByTagName("style");for(r=0;r<i.length;r++){o=i[r],a=o.getAttribute("data-href");if(a===t||a===e)return o}},r=function(r){return new Promise((function(o,a){var i=n.miniCssF(r),u=n.p+i;if(e(i,u))return o();t(r,u,o,a)}))},o={143:0};n.f.miniCss=function(t,e){var n={191:1,359:1,854:1};o[t]?e.push(o[t]):0!==o[t]&&n[t]&&e.push(o[t]=r(t).then((function(){o[t]=0}),(function(e){throw delete o[t],e})))}}(),function(){var t={143:0};n.f.j=function(e,r){var o=n.o(t,e)?t[e]:void 0;if(0!==o)if(o)r.push(o[2]);else{var a=new Promise((function(n,r){o=t[e]=[n,r]}));r.push(o[2]=a);var i=n.p+n.u(e),u=new Error,c=function(r){if(n.o(t,e)&&(o=t[e],0!==o&&(t[e]=void 0),o)){var a=r&&("load"===r.type?"missing":r.type),i=r&&r.target&&r.target.src;u.message="Loading chunk "+e+" failed.\n("+a+": "+i+")",u.name="ChunkLoadError",u.type=a,u.request=i,o[1](u)}};n.l(i,c,"chunk-"+e,e)}},n.O.j=function(e){return 0===t[e]};var e=function(e,r){var o,a,i=r[0],u=r[1],c=r[2],s=0;if(i.some((function(e){return 0!==t[e]}))){for(o in u)n.o(u,o)&&(n.m[o]=u[o]);if(c)var f=c(n)}for(e&&e(r);s<i.length;s++)a=i[s],n.o(t,a)&&t[a]&&t[a][0](),t[a]=0;return n.O(f)},r=self["webpackChunkantgo_web"]=self["webpackChunkantgo_web"]||[];r.forEach(e.bind(null,0)),r.push=e.bind(null,r.push.bind(r))}();var r=n.O(void 0,[998],(function(){return n(839)}));r=n.O(r)})();
//# sourceMappingURL=app.e92a2fc1.js.map
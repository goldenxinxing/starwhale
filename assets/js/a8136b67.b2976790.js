"use strict";(self.webpackChunkstarwhale_docs=self.webpackChunkstarwhale_docs||[]).push([[752],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return p}});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=a.createContext({}),c=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},u=function(e){var t=c(e.components);return a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},m=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,o=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),m=c(n),p=r,h=m["".concat(s,".").concat(p)]||m[p]||d[p]||o;return n?a.createElement(h,l(l({ref:t},u),{},{components:n})):a.createElement(h,l({ref:t},u))}));function p(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var o=n.length,l=new Array(o);l[0]=m;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i.mdxType="string"==typeof e?e:r,l[1]=i;for(var c=2;c<o;c++)l[c]=n[c];return a.createElement.apply(null,l)}return a.createElement.apply(null,n)}m.displayName="MDXCreateElement"},9724:function(e,t,n){n.r(t),n.d(t,{assets:function(){return u},contentTitle:function(){return s},default:function(){return p},frontMatter:function(){return i},metadata:function(){return c},toc:function(){return d}});var a=n(7462),r=n(3366),o=(n(7294),n(3905)),l=["components"],i={title:"Starwhale Client User Guide"},s=void 0,c={unversionedId:"standalone/client_user_guide",id:"standalone/client_user_guide",title:"Starwhale Client User Guide",description:"This guide explains all commands provided by the Starwhale client (swcli).",source:"@site/docs/standalone/client_user_guide.md",sourceDirName:"standalone",slug:"/standalone/client_user_guide",permalink:"/docs/standalone/client_user_guide",draft:!1,editUrl:"https://github.com/star-whale/starwhale/tree/main/docs/docs/standalone/client_user_guide.md",tags:[],version:"current",frontMatter:{title:"Starwhale Client User Guide"},sidebar:"mainSidebar",previous:{title:"main",permalink:"/docs/standalone/main"},next:{title:"standalone installing",permalink:"/docs/standalone/installation"}},u={},d=[{value:"Definitions",id:"definitions",level:2},{value:"Resource URI",id:"resource-uri",level:3},{value:"Instance URI",id:"instance-uri",level:3},{value:"Project URI",id:"project-uri",level:3},{value:"Other Resources URI",id:"other-resources-uri",level:3},{value:"Names",id:"names",level:3},{value:"Name uniqueness requirement",id:"name-uniqueness-requirement",level:4},{value:"Instance management",id:"instance-management",level:2},{value:"Select the default instance",id:"select-the-default-instance",level:3},{value:"Project management",id:"project-management",level:2},{value:"Select the default project",id:"select-the-default-project",level:3},{value:"Create a project",id:"create-a-project",level:3},{value:"List projects",id:"list-projects",level:3},{value:"Remove a project",id:"remove-a-project",level:3},{value:"Recover a project",id:"recover-a-project",level:3},{value:"Job Management",id:"job-management",level:2},{value:"List jobs",id:"list-jobs",level:3},{value:"Get a job&#39;s detailed information",id:"get-a-jobs-detailed-information",level:3},{value:"Pause a job",id:"pause-a-job",level:3},{value:"Resume a job",id:"resume-a-job",level:3},{value:"Cancel a job",id:"cancel-a-job",level:3},{value:"Utilities",id:"utilities",level:2},{value:"Garbage collection",id:"garbage-collection",level:3},{value:"Others",id:"others",level:2},{value:"Special rules for versioned resources",id:"special-rules-for-versioned-resources",level:3},{value:"Versioned resource creation and update",id:"versioned-resource-creation-and-update",level:4},{value:"Versioned resource removal and recovery",id:"versioned-resource-removal-and-recovery",level:4}],m={toc:d};function p(e){var t=e.components,n=(0,r.Z)(e,l);return(0,o.kt)("wrapper",(0,a.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"This guide explains all commands provided by the Starwhale client (swcli)."),(0,o.kt)("h2",{id:"definitions"},"Definitions"),(0,o.kt)("h3",{id:"resource-uri"},"Resource URI"),(0,o.kt)("p",null,"Resource URI is widely used in Starwhale client commands. The URI can refer to a resource in the local instance or any other resource in a remote instance. In this way, the Starwhale client can easily manipulate any resource."),(0,o.kt)("h3",{id:"instance-uri"},"Instance URI"),(0,o.kt)("p",null,"Instance URI can be either:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"A URL in format: ",(0,o.kt)("inlineCode",{parentName:"li"},"[http(s)://]<hostname or ip>[:<port>]"),", which refers to a Starwhale controller. The default scheme is HTTP, and the default port is 7827."),(0,o.kt)("li",{parentName:"ul"},"local, which means the local standalone instance.")),(0,o.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"Caveat")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},'"local" is different from "localhost". The former means the local standalone instance without a controller, while the latter implies a controller listening at the default port 7827 on localhost.|'))),(0,o.kt)("h3",{id:"project-uri"},"Project URI"),(0,o.kt)("p",null,"Project URI is in the format ",(0,o.kt)("inlineCode",{parentName:"p"},"[<Instance URI>/project/]<project name>"),". If the instance URI is not specified, use the default instance instead."),(0,o.kt)("h3",{id:"other-resources-uri"},"Other Resources URI"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Model: ",(0,o.kt)("inlineCode",{parentName:"li"},"[<Project URI>/model/]<model name>[/version/<version id>]")),(0,o.kt)("li",{parentName:"ul"},"Dataset: ",(0,o.kt)("inlineCode",{parentName:"li"},"[<Project URI>/dataset/]<dataset name>[/version/<version id>]")),(0,o.kt)("li",{parentName:"ul"},"Runtime: ",(0,o.kt)("inlineCode",{parentName:"li"},"[<Project URI>/runtime/]<runtime name>[/version/<version id>]")),(0,o.kt)("li",{parentName:"ul"},"Job: ",(0,o.kt)("inlineCode",{parentName:"li"},"[<Project URI>/job/]<job id>"))),(0,o.kt)("p",null,"If the project URI is not specified, use the default project."),(0,o.kt)("p",null,"If the version id is not specified, use the latest version."),(0,o.kt)("h3",{id:"names"},"Names"),(0,o.kt)("p",null,"Names mean project names, model names, dataset names, runtime names, and tag names."),(0,o.kt)("p",null,"Names are case-insensitive."),(0,o.kt)("p",null,"A name MUST only consist of letters ",(0,o.kt)("inlineCode",{parentName:"p"},"A-Z a-z"),", digits ",(0,o.kt)("inlineCode",{parentName:"p"},"0-9"),", the hyphen character ",(0,o.kt)("inlineCode",{parentName:"p"},"-"),", and the underscore character ",(0,o.kt)("inlineCode",{parentName:"p"},"_"),"."),(0,o.kt)("p",null,"A name should always start with a letter or the ",(0,o.kt)("inlineCode",{parentName:"p"},"_")," character."),(0,o.kt)("p",null,"The maximum length of a name is 80."),(0,o.kt)("h4",{id:"name-uniqueness-requirement"},"Name uniqueness requirement"),(0,o.kt)("p",null,"The resource name should be a unique string within its owner. For example, the project name should be unique in the owner instance, and the model name should be unique in the owner project."),(0,o.kt)("p",null,'The resource name can not be used by any other resource of the same kind in the owner, including those removed ones. For example, Project "apple" can not have two models named "Alice", even if one of them is already removed.'),(0,o.kt)("p",null,'Different kinds of resources can have the same name. For example, a project and a model can have the same name "Alice".'),(0,o.kt)("p",null,'Resources with different owners can have the same name. For example, a model in project "Apple" and a model in project "Banana" can have the same name "Alice".'),(0,o.kt)("p",null,'Garbage collected resources\' names can be reused. For example, after the model with the name "Alice" in project "Apple" is removed and garbage collected, the project can have a new model with the same name "Alice".'),(0,o.kt)("h2",{id:"instance-management"},"Instance management"),(0,o.kt)("h3",{id:"select-the-default-instance"},"Select the default instance"),(0,o.kt)("p",null,"This command sets the default Starwhale instance used by other commands."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli instance select <instance uri>\n")),(0,o.kt)("h2",{id:"project-management"},"Project management"),(0,o.kt)("h3",{id:"select-the-default-project"},"Select the default project"),(0,o.kt)("p",null,"This command sets both the default instance and project used by other commands. When a project is selected as the default project, its owner instance is also selected as the default instance."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli project select <project uri>\n")),(0,o.kt)("h3",{id:"create-a-project"},"Create a project"),(0,o.kt)("p",null,"This command creates a new project."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli project create <project uri>\n")),(0,o.kt)("h3",{id:"list-projects"},"List projects"),(0,o.kt)("p",null,"This command lists all viewable projects in the instance."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli project list [OPTIONS] [instance uri]\n")),(0,o.kt)("p",null,"Options:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"-a, --all Include removed projects which have not been garbage collected.")),(0,o.kt)("h3",{id:"remove-a-project"},"Remove a project"),(0,o.kt)("p",null,"This command removes a project."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli project remove <project uri>\n")),(0,o.kt)("h3",{id:"recover-a-project"},"Recover a project"),(0,o.kt)("p",null,"This command recovers a removed project."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli project recover <project uri>\n")),(0,o.kt)("h2",{id:"job-management"},"Job Management"),(0,o.kt)("h3",{id:"list-jobs"},"List jobs"),(0,o.kt)("p",null,"This command lists all jobs in the project."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli job list [project uri]\n")),(0,o.kt)("h3",{id:"get-a-jobs-detailed-information"},"Get a job's detailed information"),(0,o.kt)("p",null,"This command shows detailed information about a job."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli job info <job uri>\n")),(0,o.kt)("h3",{id:"pause-a-job"},"Pause a job"),(0,o.kt)("p",null,"This command pauses a running job."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli job pause <job uri>\n")),(0,o.kt)("h3",{id:"resume-a-job"},"Resume a job"),(0,o.kt)("p",null,"This command resumes a paused job."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli job resume <job uri>\n")),(0,o.kt)("h3",{id:"cancel-a-job"},"Cancel a job"),(0,o.kt)("p",null,"This command cancels a running/paused job."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli job cancel <job uri>\n")),(0,o.kt)("h2",{id:"utilities"},"Utilities"),(0,o.kt)("h3",{id:"garbage-collection"},"Garbage collection"),(0,o.kt)("p",null,"This command purges removed entities in the instance. Purged entities are not recoverable."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli gc [instance uri]\n")),(0,o.kt)("p",null,"The garbage collection aims to hold the storage size within an acceptable range. It keeps removing the oldest removed entity until the total storage size is not greater than a predefined threshold."),(0,o.kt)("h2",{id:"others"},"Others"),(0,o.kt)("h3",{id:"special-rules-for-versioned-resources"},"Special rules for versioned resources"),(0,o.kt)("p",null,"Some resources are versioned, like models, datasets, and runtimes. There are some special rules for commands manipulating these resources."),(0,o.kt)("h4",{id:"versioned-resource-creation-and-update"},"Versioned resource creation and update"),(0,o.kt)("p",null,"The command for creation and update has the following pattern:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli model/dataset/runtime build <resource uri> [working dir]\n")),(0,o.kt)("p",null,"A new resource with one version is automatically created when the specified resource URI does not exist. Otherwise, a new version of the resource is created and becomes the latest version."),(0,o.kt)("h4",{id:"versioned-resource-removal-and-recovery"},"Versioned resource removal and recovery"),(0,o.kt)("p",null,"The commands for removal and recovery have the following pattern:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-console"},"swcli model/dataset/runtime remove/recover <resource uri>\n")),(0,o.kt)("p",null,"Only the specified version will be removed/recovered if the resource URI has the version part. Otherwise, the resource with the whole history will be removed/recovered."),(0,o.kt)("p",null,"If the resource has only one version and is being removed, the resource is removed as well."))}p.isMDXComponent=!0}}]);
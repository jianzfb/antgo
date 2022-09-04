/**
 * Created by jian on 07/25/18.
 */

String.prototype.replaceAll = function(exp, newStr){
    return this.replace(new RegExp(exp, 'gm'), newStr)
};
String.prototype.format = function(args) {
    var result = this;
    if (arguments.length < 1) {
        return result;
    }

    var data = arguments; // 如果模板参数是数组
    if (arguments.length == 1 && typeof (args) == "object") {
        // 如果模板参数是对象
        data = args;
    }
    for ( var key in data) {
        var value = data[key];
        if (undefined != value) {
            result = result.replaceAll("\\{" + key + "\\}", value);
        }
    }
    return result;
};

function checkUrl(urlString){
    if(urlString!=""){
        var reg=/(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?/;
        if(!reg.test(urlString)){
                return false;
            }

            return true;
        }
    return false;
}
function checkEmail(address) {
    if(address != ""){
        var reg=/^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$/;
        if(!reg.test(address)){
            return false
        }
        return true
    }
    return false
}
function checkEnglish(english_str) {
    if(english_str != ""){
        var reg=/^[A-Za-z]+$/;
        if(!reg.test(english_str)){
            return false
        }
        return true
    }
    return false;
}
function checkChinese(chinese_str){
    if(chinese_str != ""){
        var reg=/^[\u0391-\uFFE5]+$/;
        if(!reg.test(chinese_str)){
            return false
        }
        return true
    }
    return false
}

function checkTelephone(telephone_str){
    if(telephone_str != ""){
        var reg=/^((\(\d{3}\))|(\d{3}\-))?13\d{9}$/;
        if(!reg.test(telephone_str)){
            return false
        }
        return true
    }
    return false
}


function LayoutTable(id, rows, cols){
    var layout_table = {};
    var _id = id;
    var _rows = rows;
    var _cols = cols;
    var _table_bbody;

    layout_table.append = function (r,c,node) {
        var cell_id = 'table-{id}-{r}{c}'.format({id: _id, r: r, c: c});
        // 1.step clear
        $('#'+cell_id).empty();
        // 2.step append
        $('#'+cell_id).append(node);
    }
    layout_table.text = function (r, c, text){
        var cell_id = 'table-{id}-{r}{c}'.format({id: _id, r: r, c: c});
        $('#'+cell_id).text(text);
    }

    layout_table.create = function(container) {
        var table_div = $('<div id="{id}-layout"></div>'.format({id: _id}));
        var table_div_body = $('<table class="no-border"></table>');
        table_div_body.append($('<thead><tr><th>Feature Name</th><th>Value</th></tr></thead>'));
        _table_bbody = $('<tbody id="table-{id}"></tbody>'.format({id: _id}));

        for (var r = 0; r < _rows; ++r) {
            var item = $('<tr></tr>');
            for (var c = 0; c < _cols; ++c) {
                item.append('<td id="table-{id}-{r}{c}"></td>'.format({id: _id, r: r, c: c}));
            }

            _table_bbody.append(item);
        }

        table_div_body.append(_table_bbody);
        table_div.append(table_div_body);

        container.append(table_div)
    }

    layout_table.create2 = function(container, heads) {
        var table_div = $('<div id="{id}-layout"></div>'.format({id: _id}));
        var table_div_body = $('<table class="no-border"></table>');
        var title = $('<thead><tr></tr></thead>');
        for(var i in heads){
            title.append('<th>{0}</th>'.format(heads[i]))
        }
        table_div_body.append(title);
        _table_bbody = $('<tbody id="table-{id}"></tbody>'.format({id: _id}));

        for (var r = 0; r < _rows; ++r) {
            var item = $('<tr></tr>');
            for (var c = 0; c < _cols; ++c) {
                item.append('<td id="table-{id}-{r}{c}"></td>'.format({id: _id, r: r, c: c}));
            }

            _table_bbody.append(item);
        }

        table_div_body.append(_table_bbody);
        table_div.append(table_div_body);

        container.append(table_div)
    }

    layout_table.extend_rows = function(rows){
        for (var r=0; r < rows; ++r){
            var item = $('<tr></tr>');
            for (var c = 0; c < _cols; ++c) {
                item.append('<td id="table-{id}-{r}{c}"></td>'.format({id: _id, r: _rows + r, c: c}));
            }

            _table_bbody.append(item);
        }

        _rows += rows;
    }

    return layout_table
}

function demoUserBrowser(ok_callback, bind_elem, file_filter){
    var browser_str = '<form id="demo_browser_form" name="form" action="/antgo/api/demo/submit/" method="POST" enctype="multipart/form-data"> <input id="demo_file_browser" name="file" type="file" accept="{0}"></form>'.format(file_filter)
    $("body").append($(browser_str))

    $('#demo_file_browser').change(function(e){
        $("#demo_browser_form").ajaxSubmit(
            {
                error: function(data){
                    var demo_response = eval('('+data['responseText']+')');
                    console.log(demo_response['code']+":  "+demo_response['message'])
                    alert(demo_response['code']+":  "+demo_response['message'])
                },
                success: function(data) {
                    if (ok_callback != null){
                        ok_callback(data)
                    }
            }
        })
    })

    var demo_browser_obj = {}
    demo_browser_obj.trigger = function (){
        // trigger file select browser
        $('#demo_file_browser').click()
    }

    $('#'+bind_elem).on('click', function(){
        demo_browser_obj.trigger()
    })

    return demo_browser_obj;
}

function demoMultiUserBrowser(ok_callback, trigger_elem, bind_elems, file_filters){
    var browser_str = '<form id="demo_browser_form_{0}" name="form" action="/antgo/api/demo/submit/" method="POST" enctype="multipart/form-data"></form>'.format(trigger_elem);
    var browser_obj = $(browser_str)
    $("body").append(browser_obj)
    for(var elem_index in bind_elems){
        browser_obj.append($('<input id="demo_file_browser_{0}" name="file" type="file" accept="{1}">'.format(bind_elems[elem_index], file_filters[elem_index])))
    }

    var placeholder_flags = {}

    for(var elem_index in bind_elems){
        $('#'+bind_elems[elem_index]).attr('target', bind_elems[elem_index]);
        $('#'+bind_elems[elem_index]).on('click', function(){
            var target = $(this).attr('target');
            $('#demo_file_browser_'+target).click()
        })

        placeholder_flags[bind_elems[elem_index]] = 0;

        $('#'+bind_elems[elem_index]).on('click', function(){
            var target = $(this).attr('target');
            placeholder_flags[target] = 1;
        })
    }

    $('#'+trigger_elem).on('click', function(){
        for(var index in placeholder_flags){
            if(placeholder_flags[index] == 0){
                alert('no enough data')
                return;
            }
        }

        $('#demo_browser_form_'+trigger_elem).ajaxSubmit({
            type:'post',
            url: "/antgo/api/demo/submit/",
            success: function(data){
                if(ok_callback != null){
                    ok_callback(data)
                }

                for(var elem_index in bind_elems){
                    placeholder_flags[bind_elems[elem_index]] = 0;
                }
            },
            error: function(data, XMLHttpRequest, textStatus, errorThrown){
                for(var elem_index in bind_elems){
                    placeholder_flags[bind_elems[elem_index]] = 0;
                }
                alert('fail to upload data')
                return
            }
        })
    })
}


function demoUserInput(ok_callback){
    var demo_user_input_obj = {}
    demo_user_input_obj.trigger = function(input_str){
        var is_url = checkUrl(input_str);
        var query_data = {};
        query_data['DATA'] = input_str;
        if(is_url){
            query_data['DATA_TYPE'] = 'URL';
        }
        else{
            query_data['DATA_TYPE'] = 'STRING';
        }

        $.post('/antgo/api/demo/query/', query_data, function(data, status){
            if(ok_callback != null){
                if(status == 'success'){
                    ok_callback(data)
                }
                else{
                    // do nothing
                    log.console('couldnt return processing result successfully')
                }
            }
        }).error(function(data){
            var demo_response = eval('('+data['responseText']+')');
            console.log(demo_response['code']+":  "+demo_response['message'])
            alert(demo_response['code']+":  "+demo_response['message'])
        })
    }
    return demo_user_input_obj;
}
